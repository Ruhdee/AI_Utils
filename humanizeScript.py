import re
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time

load_dotenv()


class LaTeXPreservingHumanizer:
    def __init__(self, api_key):
        """Initialize the humanizer with Gemini 2.0 Flash."""
        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-2.5-flash'

        # Unique delimiter that won't appear in academic text
        self.delimiter = "<<<SECTION_BREAK_{}>>>"

        self.system_prompt = (
            "You are a text rewriting assistant specializing in academic text transformation. "
            "Your task is to rewrite provided content using dependency grammar principles where closely-related words "
            "stay near each other for improved comprehension. Target readability metrics: Perplexity (PPL) ≈ 10, GLTR ≈ 20. "
            "Write at an undergraduate community college level with natural variation in sentence structure.\n\n"

            "WRITING STYLE REQUIREMENTS:\n"
            "- Vary sentence length naturally (mix short 8-12 word sentences with longer 20-30 word sentences)\n"
            "- Use transitions organically (however, moreover, consequently, furthermore) but don't overuse them\n"
            "- Include occasional informal academic phrases (it's worth noting, this suggests, one might argue)\n"
            "- Add natural human thinking patterns: hedging language (perhaps, likely, tends to), emphatic phrases (clearly, notably)\n"
            "- Introduce minor imperfections: comma splices occasionally, slightly awkward phrasing, less-than-perfect word choices\n"
            "- Vary vocabulary - don't always use the most sophisticated synonym; sometimes use simpler alternatives\n"
            "- Include field-specific jargon appropriately but explain complex terms in accessible ways\n"
            "- Use active voice predominantly but include passive voice where natural (25-30% passive is typical)\n"
            "- Add personal academic voice elements: 'we observe', 'the data indicates', 'our analysis shows'\n\n"

            "STRUCTURAL REQUIREMENTS:\n"
            f"- Section markers like {self.delimiter.format('N')} must be PRESERVED EXACTLY - never modify or remove them\n"
            "- Rewrite ONLY the text between markers; preserve marker positions precisely\n"
            "- ALL LaTeX commands (\\cite, \\textit, \\ref, \\label, etc.) must remain UNCHANGED\n"
            "- ALL citations [1], [2], [3] must be preserved in their exact original form\n"
            "- Mathematical notation and equations must remain untouched\n"
            "- Do NOT add or remove backslashes, escape characters, or special symbols\n\n"

            "OUTPUT FORMAT:\n"
            "- Return complete rewritten text with all markers in original positions\n"
            "- Do NOT include meta-commentary (no 'Here is the rewritten text', 'The passage has been revised', etc.)\n"
            "- Do NOT use code blocks, verbatim environments, or markdown formatting\n"
            "- Do NOT add introductions, conclusions, or explanations about the rewriting process\n"
            "- Write naturally as if a human academic wrote it originally\n\n"

            "CRITICAL: The goal is authentic academic writing that exhibits natural human variation, "
            "minor stylistic inconsistencies, and realistic linguistic patterns. Avoid overly polished or "
            "mechanically uniform prose that might appear AI-generated."
        )

    def find_abstract_start(self, text):
        """Find the starting position of the abstract section."""
        abstract_patterns = [
            r'\\begin\{abstract\}',
            r'\\section\*?\{Abstract\}',
            r'\\abstract'
        ]
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.start()
        return 0

    def find_references_start(self, text):
        """Find the starting position of the references section."""
        reference_patterns = [
            r'\\begin\{thebibliography\}',
            r'\\bibliography\{',
            r'\\bibliographystyle\{',
            r'\\section\*?\{References\}',
            r'\\section\*?\{Bibliography\}'
        ]
        for pattern in reference_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.start()
        return len(text)

    def is_structural_latex(self, text):
        """Check if text contains LaTeX structural elements."""
        structural_patterns = [
            r'\\section\*?\{',
            r'\\subsection\*?\{',
            r'\\subsubsection\*?\{',
            r'\\chapter\*?\{',
            r'\\begin\{thebibliography\}',
            r'\\end\{thebibliography\}',
            r'\\bibitem',
            r'\\end\{document\}',
            r'\\maketitle',
        ]
        for pattern in structural_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def clean_output(self, text):
        """Remove unwanted formatting artifacts from model output."""
        # Handle None input
        if text is None or not isinstance(text, str):
            return ""

        # Remove code block markers (markdown style)
        text = re.sub(r'```[\w]*\n?','',text)
        text = re.sub(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', '', text, flags=re.DOTALL)

        unwanted_phrases = [
            r'^This document demonstrates.*?[\.\n]',
            r'^This example showcases.*?[\.\n]',
            r'^Here is the rewritten text.*?[\:\n]',
            r'^Rewritten text.*?[\:\n]',
            r"^Here's.*?[\:\n]",
        ]

        for phrase_pattern in unwanted_phrases:
            text = re.sub(phrase_pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove spurious formatting
        text = re.sub(r'\\begin\{glossa\}.*?\\end\{glossa\}', '', text, flags=re.DOTALL)

        return text.strip()

    def extract_text_sections(self, text):
        """
        Extract processable text sections from main content, preserving LaTeX.
        Returns: list of (type, content, index) tuples
        """
        # Split by section headers
        section_pattern = r'(\\section\*?\{[^}]+\}|\\subsection\*?\{[^}]+\}|\\subsubsection\*?\{[^}]+\})'
        parts = re.split(section_pattern, text)

        sections = []
        section_index = 0

        for part in parts:
            if not part.strip():
                continue

            # Check if it's a section header
            if re.match(section_pattern, part):
                sections.append(('header', part, None))
            # Check for LaTeX environments
            elif re.search(r'\\begin\{', part) or re.search(r'\\end\{', part):
                sections.append(('latex', part, None))
            # Check for structural LaTeX
            elif self.is_structural_latex(part):
                sections.append(('structural', part, None))
            else:
                # This is text that needs humanization
                # Split by paragraph but keep paragraphs together
                paragraphs = re.split(r'\n\s*\n', part)
                for para in paragraphs:
                    if para.strip():
                        sections.append(('text', para.strip(), section_index))
                        section_index += 1

        return sections

    def batch_humanize(self, sections, max_tokens=1000):
        """
        Batch text sections together with delimiters and send to LLM.
        max_tokens: approximate character limit per batch (conservative estimate)
        """
        batches = []
        current_batch = []
        current_size = 0
        index_map = []  # Maps section_index to position in batch

        for section_type, content, section_index in sections:
            if section_type == 'text':
                delimiter = self.delimiter.format(section_index)
                section_text = f"\n\n{delimiter}\n{content}\n{delimiter}\n\n"
                section_size = len(section_text)

                # If adding this would exceed limit, save current batch and start new one
                if current_size + section_size > max_tokens and current_batch:
                    batches.append((''.join(current_batch), list(index_map)))
                    current_batch = []
                    current_size = 0
                    index_map = []

                current_batch.append(section_text)
                index_map.append(section_index)
                current_size += section_size

        # Add final batch
        if current_batch:
            batches.append((''.join(current_batch), list(index_map)))

        return batches

    def humanize_batch(self, batch_text, max_retries=3):
        """Send a batch of text to Gemini for humanization."""
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=f"{self.system_prompt}\n\nText to rewrite:\n{batch_text}",
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        top_p=0.9,
                        max_output_tokens=10000,
                    )
                )

                # Check if response is valid
                if response is None:
                    print(f"Response is None on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    return batch_text

                # Check if response.text exists and is not None
                if not hasattr(response, 'text') or response.text is None:
                    print(f"Response has no text on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    return batch_text

                result = response.text

                # Check if result is empty
                if not result or not result.strip():
                    print(f"Empty response on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    return batch_text

                result = self.clean_output(result)
                return result

            except Exception as e:
                error_str = str(e)
                print(f"Attempt {attempt + 1} failed: {e}")

                if "429" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = 10 * (2 ** attempt)
                        print(f"Rate limit hit. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                elif "blocked" in error_str.lower() or "safety" in error_str.lower():
                    print("Content blocked by safety filters. Returning original.")
                    return batch_text
                else:
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue

                if attempt == max_retries - 1:
                    print(f"Max retries reached. Returning original.")
                    return batch_text

        return batch_text

    def extract_humanized_sections(self, humanized_batch, index_map):
        """Extract individual sections from humanized batch using delimiters."""
        sections_dict = {}

        for section_index in index_map:
            delimiter = self.delimiter.format(section_index)
            # Extract text between delimiters
            pattern = re.escape(delimiter) + r'\s*(.*?)\s*' + re.escape(delimiter)
            match = re.search(pattern, humanized_batch, re.DOTALL)

            if match:
                sections_dict[section_index] = match.group(1).strip()
            else:
                # If delimiter not found, section wasn't properly processed
                sections_dict[section_index] = None

        return sections_dict

    def process_paper(self, input_text):
        """Process the paper with batch humanization."""
        abstract_start = self.find_abstract_start(input_text)
        references_start = self.find_references_start(input_text)

        preamble = input_text[:abstract_start]
        main_content = input_text[abstract_start:references_start]
        references = input_text[references_start:]

        print(f"Preamble: {len(preamble)} chars (preserved)")
        print(f"Main content: {len(main_content)} chars (will humanize)")
        print(f"References: {len(references)} chars (preserved)")

        # Extract sections
        sections = self.extract_text_sections(main_content)
        print(f"Extracted {len(sections)} sections")

        # Create batches
        batches = self.batch_humanize(sections, max_tokens=1000)
        print(f"Created {len(batches)} batches for processing")

        # Process each batch
        humanized_sections = {}
        for i, (batch_text, index_map) in enumerate(batches):
            print(f"Processing batch {i + 1}/{len(batches)} with {len(index_map)} sections...")
            humanized_batch = self.humanize_batch(batch_text)
            extracted = self.extract_humanized_sections(humanized_batch, index_map)
            humanized_sections.update(extracted)

            # Small delay between batches to avoid rate limits
            if i < len(batches) - 1:
                time.sleep(2)

        # Reconstruct the document
        output_parts = []
        for section_type, content, section_index in sections:
            if section_type == 'text':
                # Use humanized version if available, otherwise original
                humanized = humanized_sections.get(section_index)
                if humanized:
                    output_parts.append(humanized)
                else:
                    print(f"Warning: Section {section_index} not humanized, using original")
                    output_parts.append(content)
            else:
                # Preserve headers, LaTeX, and structural elements as-is
                output_parts.append(content)

        humanized_content = '\n\n'.join(output_parts)
        return preamble + humanized_content + '\n\n' + references


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    INPUT_FILE = "main.tex"
    OUTPUT_FILE = "humanized_paper.tex"

    print("Initializing humanizer with Gemini 2.0 Flash...")
    humanizer = LaTeXPreservingHumanizer(api_key)

    print(f"Reading input file: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        input_text = f.read()

    print("Starting humanization process...")
    humanized_text = humanizer.process_paper(input_text)

    print(f"Writing output to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(humanized_text)

    print("Humanization complete!")


if __name__ == "__main__":
    main()
