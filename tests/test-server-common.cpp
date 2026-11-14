// oaicompat_chat_params_parse: per-request thinking control.
//
// The server default (opt.enable_thinking) applies when the request is silent.
// A request can override it two ways: the OpenAI-standard "reasoning_effort"
// ("none" turns thinking off) and the vendor top-level "enable_thinking"
// boolean that Qwen/DeepSeek clients send. Both must reach the rendered
// prompt: for a thinking-capable template, "off" renders the empty
//  block that tells the model not to think.

#undef NDEBUG
#include "chat.h"
#include "json.h"
#include "server-chat.h"
#include "server-common.h"

#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using json = common_json;

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    assert(f);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Prompt of a generation request under the Qwen3 template: thinking OFF
// renders the empty  block right after the assistant header.
static bool prompt_thinking_off(const json & llama_params) {
    return llama_params.at("prompt").get<std::string>().find("\n\n\u003c/think") != std::string::npos;
}

static json parse_request(const server_chat_params & opt, json body) {
    std::vector<raw_buffer> files;
    common_chat_session session;
    return oaicompat_chat_params_parse(nullptr, body, opt, files, session);
}

int main(void) {
    server_chat_params opt;
    opt.use_jinja           = true;
    opt.prefill_assistant = false;
    opt.reasoning_format  = COMMON_REASONING_FORMAT_DEEPSEEK;
    opt.tmpls             = common_chat_templates_init(nullptr, read_file("models/templates/Qwen-Qwen3-0.6B.jinja"));
    assert(opt.tmpls);
    opt.enable_thinking = true; // the server default

    json msg = {
        { "role",    "user"    },
        { "content", "hi"      },
    };

    // server default: thinking on, no empty-think block in the prompt
    {
        json body = { { "messages", json::array({ msg }) } };
        assert(!prompt_thinking_off(parse_request(opt, body)));
    }

    // top-level "enable_thinking": false -- the wire field Qwen/DeepSeek
    // clients (and omp's title requests) send; must turn thinking off
    {
        json body = {
            { "messages",        json::array({ msg }) },
            { "enable_thinking", false },
        };
        assert(prompt_thinking_off(parse_request(opt, body)));
    }

    // top-level "enable_thinking": true -- explicit, keeps the default
    {
        json body = {
            { "messages",        json::array({ msg }) },
            { "enable_thinking", true },
        };
        assert(!prompt_thinking_off(parse_request(opt, body)));
    }

    // chat_template_kwargs override (pre-existing)
    {
        json body = {
            { "messages",            json::array({ msg }) },
            { "chat_template_kwargs", { { "enable_thinking", false } } },
        };
        assert(prompt_thinking_off(parse_request(opt, body)));
    }

    // "reasoning_effort": "none" (pre-existing; also the /v1/responses path)
    {
        json body = {
            { "messages",        json::array({ msg }) },
            { "reasoning_effort", "none" },
        };
        assert(prompt_thinking_off(parse_request(opt, body)));
    }

    // /v1/responses conversion must carry the field into the chatcmpl body
    {
        json body = {
            { "model",           "x" },
            { "input",         "hi" },
            { "enable_thinking", false },
        };
        auto converted = server_chat_convert_responses_to_chatcmpl(body);
        assert(converted.contains("enable_thinking"));
        assert(converted.at("enable_thinking").get<bool>() == false);
    }

    // conflict: the standard field wins over the vendor one
    {
        json body = {
            { "messages",         json::array({ msg }) },
            { "enable_thinking", true },
            { "reasoning_effort", "none" },
        };
        assert(prompt_thinking_off(parse_request(opt, body)));
    }

    printf("test_server_common: ok\n");
    return 0;
}
