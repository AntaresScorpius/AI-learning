from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.

def get_gemini_flash_response(prompt):
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    print(response.text)
    return response.text

def get_gemini_flash_lite_response(prompt):
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
    )
    print(response.text)
    return response.text

def compare_llm_responses(response1, response2):
    client = genai.Client(api_key=GEMINI_API_KEY)
    comparison_prompt = (
        "Compare the following two AI model responses.\n"
        "Response 1: " + response1 + "\n"
        "Response 2: " + response2 + "\n"
        "Provide a detailed comparison."
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=comparison_prompt
    )
    return response.text   

def main():
    print("=== LLM Comparison Project ===")
    prompt = "Explain how AI works in 200 words."

    print("\nQuerying Gemini 2.5 Flash...")
    flash_response = get_gemini_flash_response(prompt)
    print("\nQuerying Gemini 2.5 Flash Lite...")
    flash_lite_response = get_gemini_flash_lite_response(prompt)
    comparison_result = compare_llm_responses(flash_response, flash_lite_response)

    print("\n--- Gemini 2.5 Flash Response ---\n", flash_response)
    print("\n--- Gemini 2.5 Flash Lite Response ---\n", flash_lite_response)

    # Prepare for Markdown report
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(f"# LLM Comparison Report\n\n")
        f.write(f"## Prompt\n\n{prompt}\n\n")
        f.write(f"## Gemini 2.5 Flash Response\n\n{flash_response}\n\n")
        f.write(f"## Gemini 2.5 Flash Lite Response\n\n{flash_lite_response}\n\n")
        f.write(f"## Comparison Notes\n\n- {comparison_result}")
    print("\nReport saved to report.md")

if __name__ == "__main__":
    main()