import { anthropic, createAnthropic } from "@ai-sdk/anthropic";
import { createCohere } from "@ai-sdk/cohere";
import { createMistral } from "@ai-sdk/mistral";
import { createOpenAI, openai } from "@ai-sdk/openai";
import { ollama } from "ollama-ai-provider";

import { env } from "@/env";
import { calculateCost, getModelCost } from "@/lib/cost-calculator";
import { getDefaultConfiguration, insertLog } from "@/lib/db";
import { generateText, streamText } from "ai";
import { type NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openaiClient = new OpenAI({
  apiKey: env.OPENAI_API_KEY,
});

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

function transformCursorMessages(messages: any[]): any[] {
  if (!Array.isArray(messages)) {
    console.warn('Messages is not an array, returning empty array');
    return [];
  }
  
  return messages.map((message) => {
    if (!message || typeof message !== 'object') {
      console.warn('Invalid message object, skipping:', message);
      return null;
    }
    
    // Handle tool role messages from Cursor
    if (message.role === "tool") {
      // Transform tool messages to assistant messages with tool results
      // For Anthropic, we should not include tool_call_id or name in the main message
      return {
        role: "assistant",
        content: `Tool response (${message.name || 'unknown'}): ${message.content || ""}`,
      };
    }
    
    // Handle complex content arrays (e.g., from Claude's thinking feature)
    let transformedContent = message.content;
    if (Array.isArray(message.content)) {
      // Extract text content from content arrays, filtering out thinking and other unsupported types
      const textParts = message.content
        .filter((item: any) => item.type === "text" || item.type === "tool_result")
        .map((item: any) => {
          if (item.type === "text") {
            return item.text;
          } else if (item.type === "tool_result") {
            // Handle nested content in tool_result
            const toolContent = Array.isArray(item.content)
              ? item.content.map((c: any) => c.text || '').join('\n')
              : item.content;
            return `Tool result (${item.tool_use_id || 'unknown'}): ${toolContent}`;
          }
          return '';
        })
        .filter(Boolean)
        .join('\n\n');
      
      // If we have tool_use items, we might need to handle them differently
      const toolUses = message.content.filter((item: any) => item.type === "tool_use");
      if (toolUses.length > 0 && message.role === "assistant") {
        // This is an assistant message with tool calls
        return {
          role: "assistant",
          content: textParts || "",
          // Note: We're not passing tool_calls for now as they need special formatting
        };
      }
      
      transformedContent = textParts || "";
    }
    
    // Handle assistant messages with tool_calls
    if (message.role === "assistant" && message.tool_calls) {
      return {
        role: "assistant",
        content: transformedContent || "",
        // Note: tool_calls might need different handling for Anthropic
        tool_calls: message.tool_calls,
      };
    }
    
    // Ensure only valid roles are passed
    const validRoles = ["system", "user", "assistant"];
    if (!validRoles.includes(message.role)) {
      console.warn(`Invalid role "${message.role}", defaulting to "user"`);
      return {
        role: "user",
        content: transformedContent || "",
      };
    }
    
    // Pass through other messages as-is, but ensure they have required fields
    return {
      role: message.role,
      content: transformedContent || "",
      ...(message.name && message.role === "system" && { name: message.name }),
    };
  }).filter(Boolean); // Remove any null entries
}

function transformTools(openaiTools: any[]): Record<string, any> | undefined {
  if (!Array.isArray(openaiTools) || openaiTools.length === 0) {
    return undefined;
  }
  
  try {
    const transformedTools: Record<string, any> = {};
    
    for (const tool of openaiTools) {
      if (tool?.type === "function" && tool?.function?.name) {
        // Create a proper CoreTool structure for AI SDK
        transformedTools[tool.function.name] = {
          description: tool.function.description || "",
          parameters: {
            type: "object",
            properties: tool.function.parameters?.properties || {},
            required: tool.function.parameters?.required || [],
            ...tool.function.parameters
          },
        };
      }
    }
    
    return Object.keys(transformedTools).length > 0 ? transformedTools : undefined;
  } catch (error) {
    console.error("Error transforming tools:", error);
    return undefined;
  }
}

async function getAIModelClient(provider: string, model: string) {
  if (!provider || !model) {
    throw new Error("Provider and model are required");
  }

  switch (provider.toLowerCase()) {
    case "openai":
      if (!env.OPENAI_API_KEY) {
        throw new Error("OpenAI API key is not configured");
      }
      return openai(model);
    case "anthropic": {
      if (!env.ANTHROPIC_API_KEY) {
        throw new Error("Anthropic API key is not configured");
      }
      const anthropicClient = createAnthropic({
        apiKey: env.ANTHROPIC_API_KEY,
      });
      return anthropicClient(model);
    }
    case "anthropiccached": {
      if (!env.ANTHROPIC_API_KEY) {
        throw new Error("Anthropic API key is not configured");
      }
      const anthropicClient = createAnthropic({
        apiKey: env.ANTHROPIC_API_KEY,
      });
      return anthropicClient(model, { cacheControl: true });
    }
    case "cohere": {
      if (!env.COHERE_API_KEY) {
        throw new Error("Cohere API key is not configured");
      }
      const cohereClient = createCohere({
        apiKey: env.COHERE_API_KEY,
      });
      return cohereClient(model);
    }
    case "mistral": {
      if (!env.MISTRAL_API_KEY) {
        throw new Error("Mistral API key is not configured");
      }
      const mistralClient = createMistral({
        apiKey: env.MISTRAL_API_KEY,
      });
      return mistralClient(model);
    }
    case "groq": {
      if (!env.GROQ_API_KEY) {
        throw new Error("Groq API key is not configured");
      }
      const groqClient = createOpenAI({
        apiKey: env.GROQ_API_KEY,
      });
      return groqClient(model);
    }
    case "ollama":
      return ollama("llama3.1");
    case "google-vertex":
      throw new Error("Google Vertex AI is not currently supported");
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: { openai: string[] } },
) {
  const startTime = Date.now(); // Track conversation start time
  const endpoint = params.openai.join("/");
  console.log("POST request received:", {
    endpoint,
    url: request.url,
    headers: Object.fromEntries(request.headers),
  });

  if (endpoint !== "chat/completions" && endpoint !== "v1/chat/completions") {
    return NextResponse.json({ error: "Not found", endpoint }, { status: 404 });
  }

  const body = await request.json();
  console.log("Request body:", JSON.stringify(body, null, 2));
  const { messages, model: cursorModel, stream = false, tools, ...otherParams } = body;

  try {
    const defaultConfig = await getDefaultConfiguration();
    if (!defaultConfig) {
      throw new Error("No default configuration found");
    }

    const {
      id: configId,
      provider,
      model,
      temperature,
      maxTokens,
      topP,
      frequencyPenalty,
      presencePenalty,
    } = defaultConfig;

    if (!provider || typeof provider !== 'string') {
      throw new Error("Provider is not defined in the default configuration");
    }

    if (!model || typeof model !== 'string') {
      throw new Error("Model is not defined in the default configuration");
    }

    const aiModel = await getAIModelClient(provider, model);

    // Validate and transform messages
    if (!messages || !Array.isArray(messages)) {
      throw new Error("Invalid messages format");
    }

    // Transform Cursor messages to AI SDK format
    console.log("Original messages:", JSON.stringify(messages, null, 2));
    let modifiedMessages = transformCursorMessages(messages);
    console.log("Transformed messages:", JSON.stringify(modifiedMessages, null, 2));
    console.log("Provider:", provider, "Model:", model);
    
    if (modifiedMessages.length === 0) {
      throw new Error("No valid messages found after transformation");
    }

    // Transform tools from OpenAI array format to AI SDK Record format
    console.log("Tools parameter received:", tools);
    
    // Temporarily disable tools for GPT models to debug the core issue
    const validatedTools = provider.toLowerCase() === "openai" ? undefined : transformTools(tools);

    if (provider.toLowerCase() === "anthropiccached") {
      const hasPotentialContext = modifiedMessages.some(
        (message: any) => message.name === "potential_context",
      );

      modifiedMessages = modifiedMessages.map((message: any) => {
        if (message.name === "potential_context") {
          return {
            ...message,
            experimental_providerMetadata: {
              anthropic: { cacheControl: { type: "ephemeral" } },
            },
          };
        }
        return message;
      });

      if (!hasPotentialContext && modifiedMessages.length >= 2) {
        modifiedMessages[1] = {
          ...modifiedMessages[1],
          experimental_providerMetadata: {
            anthropic: { cacheControl: { type: "ephemeral" } },
          },
        };
      }
    }

    const streamTextOptions = {
      model: aiModel,
      messages: modifiedMessages,
      maxTokens: ["anthropic", "anthropiccached"].includes(
        provider.toLowerCase(),
      )
        ? 8192
        : undefined,
      ...(validatedTools && { tools: validatedTools }),
    };

    const logEntry = {
      method: "POST",
      url: `/api/${endpoint}`,
      headers: Object.fromEntries(request.headers),
      body: {
        ...body,
        ...streamTextOptions,
        model: model,
      },
      response: {},
      timestamp: new Date(),
      metadata: {
        configId,
        provider,
        model,
        temperature,
        maxTokens,
        topP,
        frequencyPenalty,
        presencePenalty,
        inputTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
        inputCost: 0,
        outputCost: 0,
        totalCost: 0,
      },
    };

    if (stream) {
      const result = await streamText({
        ...streamTextOptions,
        async onFinish({
          text,
          toolCalls,
          toolResults,
          usage,
          finishReason,
          ...otherProps
        }) {
          const inputTokens = usage?.promptTokens ?? 0;
          const outputTokens = usage?.completionTokens ?? 0;
          const totalTokens = usage?.totalTokens ?? 0;

          const modelCost = await getModelCost(provider, model);
          const inputCost = (inputTokens / 1000000) * modelCost.inputTokenCost;
          const outputCost =
            (outputTokens / 1000000) * modelCost.outputTokenCost;
          const totalCost = inputCost + outputCost;

          console.log('Streaming onFinish - toolCalls:', toolCalls);
          console.log('Streaming onFinish - toolResults:', toolResults);

          logEntry.response = {
            text,
            toolCalls,
            toolResults,
            usage,
            finishReason,
            ...otherProps,
          };
          logEntry.metadata = {
            ...logEntry.metadata,
            inputTokens,
            outputTokens,
            totalTokens,
            inputCost,
            outputCost,
            totalCost,
          };
          await insertLog(logEntry);
        },
      });

      // Convert the result to a ReadableStream in OpenAI's format
      const stream = new ReadableStream({
        async start(controller) {
          for await (const chunk of result.textStream) {
            const data = JSON.stringify({
              id: `chatcmpl-${Math.random().toString(36).substr(2, 9)}`,
              object: "chat.completion.chunk",
              created: Math.floor(Date.now() / 1000),
              model: model,
              choices: [
                {
                  delta: { content: chunk },
                  index: 0,
                  finish_reason: null,
                },
              ],
            });
            controller.enqueue(new TextEncoder().encode(`data: ${data}\n\n`));
          }
          controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
          controller.close();
        },
      });

      // Return a streaming response
      return new Response(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization, x-api-key, ngrok-skip-browser-warning",
        },
      });
    }
    // For non-streaming requests, use the AI SDK
    const result = await generateText({
      model: aiModel,
      messages: modifiedMessages,
      ...(validatedTools && { tools: validatedTools }),
    });

    console.log('Non-streaming result - toolCalls:', result.toolCalls);
    console.log('Non-streaming result - toolResults:', result.toolResults);

    const inputTokens = result.usage?.promptTokens ?? 0;
    const outputTokens = result.usage?.completionTokens ?? 0;
    const totalTokens = result.usage?.totalTokens ?? 0;

    const modelCost = await getModelCost(provider, model);
    const inputCost = inputTokens * modelCost.inputTokenCost;
    const outputCost = outputTokens * modelCost.outputTokenCost;
    const totalCost = inputCost + outputCost;

    logEntry.response = result;
    logEntry.metadata = {
      ...logEntry.metadata,
      inputTokens,
      outputTokens,
      totalTokens,
      inputCost,
      outputCost,
      totalCost,
    };
    await insertLog(logEntry);

    return NextResponse.json(result, {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, x-api-key, ngrok-skip-browser-warning",
      },
    });
  } catch (error) {
    console.error("Error in chat completion:", error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    const errorLogEntry = {
      method: "POST",
      url: `/api/${endpoint}`,
      headers: Object.fromEntries(request.headers),
      body: body,
      response: { error: errorMessage },
      timestamp: new Date(),
      metadata: {
        error: errorMessage,
        stack: (error as Error).stack,
      },
    };
    await insertLog(errorLogEntry);
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: { openai: string[] } },
) {
  const endpoint = params.openai.join("/");
  console.log("GET request received:", {
    endpoint,
    url: request.url,
    headers: Object.fromEntries(request.headers),
  });

  // Handle both 'models' and 'v1/models' endpoints
  if (endpoint === "models" || endpoint === "v1/models") {
    const logEntry = {
      method: "GET",
      url: `/api/${endpoint}`,
      headers: Object.fromEntries(request.headers),
      body: {},
      response: {},
      timestamp: new Date(),
      metadata: {}, // Add empty metadata object to satisfy Prisma schema
    };

    try {
      const models = await openaiClient.models.list();
      logEntry.response = models;
      await insertLog(logEntry);
      return NextResponse.json(models, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization, x-api-key, ngrok-skip-browser-warning",
        },
      });
    } catch (error) {
      console.error("Error fetching models:", error);
      logEntry.response = { error: String(error) };
      logEntry.metadata = { error: String(error) }; // Add error to metadata
      await insertLog(logEntry);
      return NextResponse.json({ error: String(error) }, { 
        status: 500,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization, x-api-key, ngrok-skip-browser-warning",
        },
      });
    }
  }

  // New test routes
  else if (endpoint === "test/openai") {
    return testOpenAI();
  } else if (endpoint === "test/anthropic") {
    return testAnthropic();
  } else if (endpoint === "test/anthropiccached") {
    return testAnthropicCached();
  } else if (endpoint === "test/cohere") {
    return testCohere();
  } else if (endpoint === "test/mistral") {
    return testMistral();
  } else if (endpoint === "test/groq") {
    return testGroq();
  }

  // Log any unmatched endpoints
  console.log("Unmatched GET endpoint:", endpoint);
  return NextResponse.json({ error: "Not found", endpoint }, { status: 404 });
}

async function testOpenAI() {
  try {
    const model = openai("gpt-3.5-turbo");
    const result = await generateText({
      model,
      messages: [{ role: "user", content: 'Say "Hello from OpenAI!"' }],
    });
    return NextResponse.json({ provider: "OpenAI", result });
  } catch (error) {
    console.error("Error testing OpenAI:", error);
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}

async function testAnthropicCached() {
  try {
    const model = anthropic("claude-3-5-sonnet-20240620", {
      cacheControl: true,
    });

    const result = await generateText({
      model,
      messages: [
        { role: "user", content: 'Say "Hello from Anthropic and Vercel"' },
      ],
    });
    return NextResponse.json({ provider: "Anthropic Cached", result });
  } catch (error) {
    console.error("Error testing Anthropic:", error);
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}

async function testAnthropic() {
  try {
    const anthropicClient = createAnthropic({
      apiKey: env.ANTHROPIC_API_KEY,
    });
    const model = anthropicClient("claude-3-haiku-20240307");
    const result = await generateText({
      model,
      messages: [{ role: "user", content: 'Say "Hello from Anthropic!"' }],
    });
    return NextResponse.json({ provider: "Anthropic", result });
  } catch (error) {
    console.error("Error testing Anthropic:", error);
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}

async function testCohere() {
  try {
    const cohereClient = createCohere({
      apiKey: env.COHERE_API_KEY,
    });
    const model = cohereClient("command");
    const result = await generateText({
      model,
      messages: [{ role: "user", content: 'Say "Hello from Cohere!"' }],
    });
    return NextResponse.json({ provider: "Cohere", result });
  } catch (error) {
    console.error("Error testing Cohere:", error);
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}

async function testMistral() {
  try {
    const mistralClient = createMistral({
      apiKey: env.MISTRAL_API_KEY,
    });
    const model = mistralClient("mistral-small-latest");
    const result = await generateText({
      model,
      messages: [{ role: "user", content: 'Say "Hello from Mistral!"' }],
    });
    return NextResponse.json({ provider: "Mistral", result });
  } catch (error) {
    console.error("Error testing Mistral:", error);
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}

async function testGroq() {
  try {
    const groqClient = createOpenAI({
      apiKey: env.GROQ_API_KEY,
    });
    const model = groqClient("llama-3.1-70b-versatile");
    const result = await generateText({
      model,
      messages: [{ role: "user", content: 'Say "Hello from Groq!"' }],
    });
    return NextResponse.json({ provider: "Groq", result });
  } catch (error) {
    console.error("Error testing Groq:", error);
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}

// Handle OPTIONS requests for CORS
export async function OPTIONS(
  request: NextRequest,
  { params }: { params: { openai: string[] } },
) {
  const endpoint = params.openai.join("/");
  console.log("OPTIONS request received:", {
    endpoint,
    url: request.url,
    headers: Object.fromEntries(request.headers),
  });

  return new NextResponse(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, x-api-key, ngrok-skip-browser-warning",
      "Access-Control-Max-Age": "86400",
    },
  });
}
