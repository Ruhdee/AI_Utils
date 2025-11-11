import re
import os
from dotenv import load_dotenv
import ollama
import time

load_dotenv()


class LaTeXPreservingHumanizer:
    def __init__(self):
        """Initialize the humanizer with Ollama phi3:mini."""
        self.model = 'phi3:mini'

        self.system_prompt = (
            "You are a text rewriting assistant. Your ONLY job is to rewrite the provided text "
            "to improve readability using dependency grammar linguistic framework rather than phrase structure grammar for the output."
            "Keep closely-related words near each other to improve comprehension."
            "Rewrite the given academic or technical passage to achieve a Perplexity (PPL) score around 10 and a GLTR score around 20.\n\n"
            "CRITICAL RULES:\n"
            "1. Return ONLY the rewritten text - no explanations, no comments, no preambles\n"
            "2. Do NOT use verbatim blocks, code blocks, or special formatting\n"
            "3. Preserve ALL LaTeX commands EXACTLY as they appear (\\cite, \\textit, etc.)\n"
            "4. Preserve ALL citations like [1], [2], [3] EXACTLY\n"
            "5. Do NOT add backslashes or escape characters\n"
            "6. Do NOT wrap output in quotes or code blocks\n"
            "7. Write natural, flowing text only"
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

    def is_latex_heavy(self, text):
        """Check if text contains significant LaTeX formatting."""
        latex_indicators = [
            r'\\begin\{',
            r'\\end\{',
            r'\\[a-zA-Z]+\{',
            r'\$.*?\$'
        ]
        latex_count = sum(len(re.findall(pattern, text)) for pattern in latex_indicators)
        return latex_count > 3  # If more than 3 LaTeX commands, preserve as-is

    def clean_output(self, text):
        """Remove unwanted formatting artifacts from model output."""
        # Remove code block markers (markdown style)
        text = re.sub(r'```[\w]*\n?','',text)
        text = re.sub(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', '', text, flags=re.DOTALL)
        # Remove common preambles
        unwanted_phrases = [
            r'^This document demonstrates.*?[\.\n]',
            r'^This example showcases.*?[\.\n]',
            r'^Here is the rewritten text.*?[\:\n]',
            r'^Rewritten text.*?[\:\n]',
            r"^Here's.*?[\:\n]",  # Use double quotes for strings with apostrophes
            r'^\{\\it$$[0-9]+$$\}',  # Remove {\\it} patterns[1]
            r'^\{\\textsf$$[0-9]+$$\}',  # Remove {\\textsf} patterns[1]
        ]

        for phrase_pattern in unwanted_phrases:
            text = re.sub(phrase_pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove spurious formatting
        text = re.sub(r'\\begin\{glossa\}.*?\\end\{glossa\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\{\\it$$[0-9]+$$\}', '', text)
        text = re.sub(r'\{\\textsf$$[0-9]+$$\}', '', text)
        text = re.sub(r'\\textbf\{', '', text)

        # Be more selective about removing braces - don't remove ALL of them
        # Only remove the orphaned ones from formatting artifacts
        text = re.sub(r'\{\\it\}', '', text)
        text = re.sub(r'\{\\textsf\}', '', text)
        text = re.sub(r'\}\\textbf\{', '', text)

        return text.strip()

    def humanize_text_section(self, text, max_retries=2):
        """Send a text section to Ollama phi3:mini for humanization."""
        # Skip if text is too LaTeX-heavy
        if self.is_latex_heavy(text):
            print(f"Skipping LaTeX-heavy section ({len(text)} chars)...")
            return text

        for attempt in range(max_retries):
            try:
                response = ollama.generate(
                    model=self.model,
                    prompt=f"{self.system_prompt}\n\nText to rewrite:\n\n{text}\n\nRewritten text:",
                    options={
                        'temperature': 0.3,  # Lower temperature for more controlled output
                        'top_p': 0.9,
                        'num_predict': 2000,
                    }
                )

                result = response['response'].strip()
                result = self.clean_output(result)

                # If output is corrupted or too different in length, return original
                if len(result) < len(text) * 0.5 or len(result) > len(text) * 2:
                    print(f"Output length mismatch, using original text")
                    return text

                # Check for verbatim blocks in output
                if '\\begin{verbatim}' in result or '\\begin{glossa}' in result:
                    print(f"Detected formatting errors, using original text")
                    return text

                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    print(f"Error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed. Using original text. Error: {e}")
                    return text

    def split_into_sentences(self, text):
        """Split text into sentences while preserving LaTeX."""
        # Simple sentence splitter that respects LaTeX
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return sentences

    def is_structural_latex(self, text):
        """Check if text contains LaTeX structural elements that should never be humanized."""
        structural_patterns = [
            r'\\section\*?\{',
            r'\\subsection\*?\{',
            r'\\subsubsection\*?\{',
            r'\\chapter\*?\{',
            r'\\paragraph\{',
            r'\\subparagraph\{',
            r'\\begin\{thebibliography\}',
            r'\\end\{thebibliography\}',
            r'\\bibitem',
            r'\\bibliography\{',
            r'\\bibliographystyle\{',
            r'\\end\{document\}',
            r'\\maketitle',
            r'\\tableofcontents',
        ]

        for pattern in structural_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def process_paper(self, input_text, chunk_size=800):
        """Process the paper with improved section handling."""
        abstract_start = self.find_abstract_start(input_text)
        references_start = self.find_references_start(input_text)

        preamble = input_text[:abstract_start]
        main_content = input_text[abstract_start:references_start]
        references = input_text[references_start:]

        print(f"Preamble: {len(preamble)} chars (preserved)")
        print(f"Main content: {len(main_content)} chars (will humanize)")
        print(f"References: {len(references)} chars (preserved)")

        # Split by section headers to preserve them
        section_pattern = r'(\\section\*?\{[^}]+\}|\\subsection\*?\{[^}]+\}|\\subsubsection\*?\{[^}]+\})'
        parts = re.split(section_pattern, main_content)

        humanized_paragraphs = []

        for i, part in enumerate(parts):
            if not part.strip():
                continue

            # If it's a section header, preserve it exactly
            if re.match(section_pattern, part):
                print(f"Part {i + 1}: Preserving section header")
                humanized_paragraphs.append(part)
                continue

            # Check for structural LaTeX
            if self.is_structural_latex(part):
                print(f"Part {i + 1}: Preserving structural LaTeX")
                humanized_paragraphs.append(part)
                continue

            # Skip LaTeX environment blocks entirely
            if re.search(r'\\begin\{', part) or re.search(r'\\end\{', part):
                print(f"Part {i + 1}: Preserving LaTeX environment")
                humanized_paragraphs.append(part)
                continue

            # Process text content in paragraphs
            paragraphs = re.split(r'\n\s*\n', part)

            for j, para in enumerate(paragraphs):
                if not para.strip():
                    continue

                # Process paragraph
                if len(para) > chunk_size:
                    sentences = self.split_into_sentences(para)
                    buffer = []
                    buffer_size = 0

                    for sent in sentences:
                        buffer.append(sent)
                        buffer_size += len(sent)

                        if buffer_size >= chunk_size:
                            combined = ' '.join(buffer)
                            print(f"Part {i + 1}.{j + 1} chunk: Humanizing {len(combined)} chars...")
                            humanized = self.humanize_text_section(combined)
                            humanized_paragraphs.append(humanized)
                            buffer = []
                            buffer_size = 0

                    if buffer:
                        combined = ' '.join(buffer)
                        print(f"Part {i + 1}.{j + 1} final: Humanizing {len(combined)} chars...")
                        humanized = self.humanize_text_section(combined)
                        humanized_paragraphs.append(humanized)
                else:
                    print(f"Part {i + 1}.{j + 1}: Humanizing {len(para)} chars...")
                    humanized = self.humanize_text_section(para)
                    humanized_paragraphs.append(humanized)

        # Reassemble with proper spacing
        humanized_content = '\n\n'.join(humanized_paragraphs)

        return preamble + humanized_content + '\n\n' + references


def main():
    INPUT_FILE = "main.tex"
    OUTPUT_FILE = "humanized_paper.tex"

    print("Initializing humanizer with Ollama phi3:mini...")
    humanizer = LaTeXPreservingHumanizer()

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
