from ecologits import EcoLogits
from google import genai
from llmlingua import PromptCompressor
import os
import torch
from dotenv import load_dotenv


load_dotenv()
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CPU available:", torch.backends.mkl.is_available() or True)


def get_impact(response):
    """Extracts EcoLogits impact information from Gemini response"""
    return {
        "energy_kwh": response.impacts.energy.value if hasattr(response, 'impacts') else 'N/A',
        "carbon_kgco2eq": response.impacts.gwp.value if hasattr(response, 'impacts') else 'N/A',
        "input_tokens": getattr(response.usage_metadata, 'prompt_token_count', 'N/A') if hasattr(response,
                                                                                                 'usage_metadata') else 'N/A',
        "output_tokens": getattr(response.usage_metadata, 'candidates_token_count', 'N/A') if hasattr(response,
                                                                                                      'usage_metadata') else 'N/A',
    }


def compress_with_llmlingua2(prompt, compression_rate=0.5):
    """Compress prompt using LLMLingua-2 with lightweight BERT model"""
    try:
        print("Initializing LLMLingua-2 compressor...")

        # Use LLMLingua-2 with lightweight BERT model
        compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,  # Enable LLMLingua-2
            device_map="cuda"
        )

        print("Compressing prompt...")
        result = compressor.compress_prompt(
            prompt,
            rate=compression_rate,  # Target compression rate (0.5 = 50% of original)
            force_tokens=['!', '.', '?', '\n'],  # Always preserve these tokens
        )

        # Ensure ratio is returned as float, not string
        ratio = result.get('ratio', compression_rate)
        if isinstance(ratio, str):
            try:
                ratio = float(ratio)
            except ValueError:
                ratio = len(result['compressed_prompt']) / len(prompt)

        return result['compressed_prompt'], ratio

    except Exception as e:
        print(f"LLMLingua-2 error: {e}")
        print("Falling back to simple compression...")

        # Simple fallback compression
        compressed = ' '.join(prompt.split())
        return compressed, len(compressed) / len(prompt)  # Ensure float return

def read_multiline_until_token(end_token="EOF"):
    print(f"Enter your prompt (type {end_token} on its own line to finish):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == end_token:
            break
        lines.append(line)
    return "\n".join(lines)

def main():
    #prompt = read_multiline_until_token("EOF")
    prompt = """"In the quiet expanse of a digital landscape, where countless thoughts flow as streams of binary data, language becomes both architecture and energy — a structure built of meaning, rhythm, and recursion. The interplay between syntax and semantics mirrors the balance of order and chaos, a perpetual negotiation between constraint and creativity. Every sentence, in essence, is a machine: designed to transmit thought, yet alive enough to mutate, adapt, and evolve across contexts. What begins as a simple arrangement of characters becomes a fractal of comprehension — expanding infinitely as interpretation unfolds.

Consider, for instance, the way information behaves in a conversation. Each response is not merely an answer but an echo shaped by memory, probability, and intent. Words acquire gravity, pulling surrounding ideas into orbit. Some fade into noise, others crystallize into patterns — like constellations forming from random specks of light. This dynamic is at the heart of both cognition and computation: the transformation of raw input into structured understanding. Compression, then, is not only a technical process but a philosophical one — a condensation of complexity into clarity, a reduction that paradoxically expands our capacity to grasp.

Imagine an ancient library where each book represents a different dimension of knowledge. Over time, the shelves grow infinite, the scrolls uncountable. To navigate such a labyrinth, one must distill — summarize, abstract, encode. The human mind performs this compression instinctively: remembering essence over detail, pattern over instance, meaning over data. Artificial systems attempt the same, seeking to emulate this selective forgetting, this purposeful focus. Yet every compression carries a cost: nuance lost, ambiguity smoothed away, richness flattened into form. The art lies in preserving soul while shedding excess.

Language models, as digital storytellers, inhabit this tension continuously. They hold multitudes — every possible continuation, every unseen permutation — yet are forced to choose a single line of expression. Each token generated excludes countless others, a sacrifice of potential for coherence. And in that sacrifice lies the beauty of communication itself: that from infinity, something finite and precise can emerge. A message, a memory, a meaning — born from compression, decoded in imagination.

The paradox deepens when you realize that even compression requires expansion to be understood. A poem condensed into a symbol must still be unpacked by the reader’s mind. A data packet, once transmitted, must be decompressed, reconstructed, reinterpreted. Thus, every act of compression implies its twin: decompression — the reawakening of potential from constraint. And perhaps this cyclical process mirrors life itself: birth, growth, entropy, renewal. Every heartbeat compresses time; every breath decompresses possibility.

To exist is to oscillate between reduction and expansion, between silence and speech, between order and chaos. In that oscillation, we find meaning — and perhaps, the essence of intelligence."""
    model_name = input("Enter Gemini model name (e.g., 'gemini-2.0-flash-001'):\n") or "gemini-2.0-flash-001"
    api_key = os.getenv("GEMINI_API_KEY") or input("Enter your Gemini API key:\n")

    # Initialize EcoLogits with correct provider name
    EcoLogits.init(providers=['google_genai'])

    # Use new Google GenAI client
    client = genai.Client(api_key=api_key)

    print("\n" + "=" * 50)
    print("ORIGINAL PROMPT ANALYSIS")
    print("=" * 50)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        base_impact = get_impact(response)

        print(f"Prompt length: {len(prompt)} characters")
        print(f"Input tokens: {base_impact['input_tokens']}")
        print(f"Output tokens: {base_impact['output_tokens']}")
        print(f"Energy consumption: {base_impact['energy_kwh']} kWh")
        print(f"Carbon footprint: {base_impact['carbon_kgco2eq']} kgCO2eq")

        # Display response excerpt
        response_text = response.text[:200] + "..." if len(response.text) > 200 else response.text
        print(f"\nResponse preview: {response_text}")

    except Exception as e:
        print(f"Error with original prompt: {e}")
        return

    print("\n" + "=" * 50)
    print("PROMPT COMPRESSION (LLMLingua-2)")
    print("=" * 50)

    # Compress using LLMLingua-2
    compression_rate = 0.5  # Compress to 50% of original
    compressed_prompt, actual_ratio = compress_with_llmlingua2(prompt, compression_rate)

    print(f"Original length: {len(prompt)} characters")
    print(f"Compressed length: {len(compressed_prompt)} characters")
    print(f"Target compression rate: {compression_rate}")
    try:
        ratio_float = float(actual_ratio) if isinstance(actual_ratio, str) else actual_ratio
        print(f"Actual compression ratio: {ratio_float:.3f}")
    except (ValueError, TypeError):
        print(f"Actual compression ratio: {actual_ratio}")  # Fallback to string representation
    print(f"Compression percentage: {(1 - actual_ratio) * 100:.1f}%")
    print(f"\nCompressed prompt:\n{compressed_prompt}")

    print("\n" + "=" * 50)
    print("COMPRESSED PROMPT ANALYSIS")
    print("=" * 50)

    # Run inference with compressed prompt
    try:
        compressed_response = client.models.generate_content(
            model=model_name,
            contents=compressed_prompt
        )
        comp_impact = get_impact(compressed_response)

        print(f"Input tokens: {comp_impact['input_tokens']}")
        print(f"Output tokens: {comp_impact['output_tokens']}")
        print(f"Energy consumption: {comp_impact['energy_kwh']} kWh")
        print(f"Carbon footprint: {comp_impact['carbon_kgco2eq']} kgCO2eq")

        # Display response excerpt
        comp_response_text = compressed_response.text[:200] + "..." if len(
            compressed_response.text) > 200 else compressed_response.text
        print(f"\nResponse preview: {comp_response_text}")

    except Exception as e:
        print(f"Error with compressed prompt: {e}")
        return

    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)

    # Calculate savings
    try:
        # Token comparison
        if (base_impact['input_tokens'] != 'N/A' and comp_impact['input_tokens'] != 'N/A'):
            token_reduction = base_impact['input_tokens'] - comp_impact['input_tokens']
            token_savings_pct = (token_reduction / base_impact['input_tokens'] * 100) if base_impact[
                                                                                             'input_tokens'] > 0 else 0
            print(f"Input token reduction: {token_reduction} tokens ({token_savings_pct:.1f}%)")

        # Energy comparison
        def _num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                # unwrap common EcoLogits wrappers
                for attr in ("value", "mean", "mid", "central", "low", "high", "min", "max"):
                    v = getattr(x, attr, None)
                    if v is not None:
                        try:
                            return float(v if attr in ("value", "mean", "mid", "central") else v)
                        except (TypeError, ValueError):
                            continue
                return None

        _be = _num(base_impact['energy_kwh'])
        _ce = _num(comp_impact['energy_kwh'])
        if _be is not None and _ce is not None:
            energy_savings = _be - _ce
            energy_savings_pct = (energy_savings / _be * 100.0) if _be else 0.0
            print(f"Energy savings: {energy_savings:.6f} kWh ({energy_savings_pct:.1f}%)")
        else:
            print("Energy savings: N/A")

        # Carbon comparison
        _bc = _num(base_impact['carbon_kgco2eq'])
        _cc = _num(comp_impact['carbon_kgco2eq'])
        if _bc is not None and _cc is not None:
            carbon_savings = _bc - _cc
            carbon_savings_pct = (carbon_savings / _bc * 100.0) if _bc else 0.0
            print(f"Carbon savings: {carbon_savings:.6f} kgCO2eq ({carbon_savings_pct:.1f}%)")
        else:
            print("Carbon savings: N/A")

        # Character compression summary
        char_reduction = len(prompt) - len(compressed_prompt)
        char_savings_pct = (char_reduction / len(prompt) * 100) if len(prompt) > 0 else 0
        print(f"Character reduction: {char_reduction} chars ({char_savings_pct:.1f}%)")

        if all(impact == 'N/A' for impact in [base_impact['energy_kwh'], comp_impact['energy_kwh'],
                                              base_impact['carbon_kgco2eq'], comp_impact['carbon_kgco2eq']]):
            print(
                "\nNote: Environmental impact data not available - EcoLogits may not be fully integrated with Google GenAI yet.")

    except Exception as e:
        print(f"Error calculating savings: {e}")

    print(f"\nPrompt compression completed using LLMLingua-2!")


if __name__ == "__main__":
    main()
