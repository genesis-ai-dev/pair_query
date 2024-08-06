import os
import re
from codes import book_codes, verses

def clean_verse_text(text):
    text = text.split("+")[0]
    # Remove USFM/SFM markers
    cleaned = re.sub(r'\\[a-z]+\s*', '', text)
    # Remove any remaining markers like \*
    cleaned = re.sub(r'\\[^\s]+\s*', '', cleaned)
    # Remove numbers
    cleaned = re.sub(r'\d+', '', cleaned)
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    cleaned = cleaned.replace("LATER", "")
    return cleaned

def get_book_number(book_name):
    for code, data in book_codes.items():
        if code == book_name:
            return data['number']
    return None  # Return None if book not found

def process_sfm_files(directory, output_file, unexpected_output_file):
    verses_content = {}
    unexpected_verses_content = {}

    # Process all .SFM files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.SFM') or filename.endswith('.sfm'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Extract book code
                book_match = re.search(r'\\id\s+(\w+)', content)
                if book_match:
                    book_code = book_match.group(1)
                    try:
                        book_number = book_codes[book_code]['number']
                    except KeyError:
                        print(f"Skipping: {book_code}")
                        continue

                    # Extract chapters and verses
                    chapter_matches = re.finditer(r'\\c\s+(\d+)(.*?)(?=\\c|\Z)', content, re.DOTALL)
                    for chapter_match in chapter_matches:
                        chapter = int(chapter_match.group(1))
                        chapter_content = chapter_match.group(2)
                        
                        verse_matches = re.finditer(r'\\v\s+(\d+)\s+(.*?)(?=\\v|\Z)', chapter_content, re.DOTALL)
                        for verse_match in verse_matches:
                            verse = int(verse_match.group(1))
                            verse_content = verse_match.group(2).strip()
                            clean_content = clean_verse_text(verse_content)
                            
                            reference = f"{book_code} {chapter}:{verse}"
                            if reference in verses:
                                verses_content[(book_number, chapter, verse)] = clean_content
                            else:
                                unexpected_verses_content[reference] = clean_content

    # Write verses to output file
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for reference in verses:
            book_name, chapter_verse = reference.split(' ')
            chapter, verse = map(int, chapter_verse.split(':'))
            book_num = get_book_number(book_name)
            
            if book_num is None:
                print(f"Skipping unknown book: {book_name}")
                continue
            
            if (book_num, chapter, verse) in verses_content:
                out_file.write(f"{reference} {verses_content[(book_num, chapter, verse)]}\n")
            else:
                out_file.write(f"{reference}\n")  # Blank line for missing verse content

    # Write unexpected verses to a separate file
    with open(unexpected_output_file, 'w', encoding='utf-8') as unexp_file:
        for reference, content in unexpected_verses_content.items():
            unexp_file.write(f"{reference} {content}\n")

    print(f"Main output written to: {output_file}")
    print(f"Unexpected verses written to: {unexpected_output_file}")

# Usage
process_sfm_files('/Users/daniellosey/Desktop/code/biblica/pair_query/Gojri/', 'output.txt', 'unexpected_verses.txt')