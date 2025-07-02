import json


def clean_json_data(input_file, output_file):
    """
    Process JSON file, keeping only question, supporting_facts and answer fields

    Args:
        input_file (str): Input file path
        output_file (str): Output file path
    """
    try:
        # Read original JSON file
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Process data, keeping only required fields
        cleaned_data = []

        for item in data:
            cleaned_item = {
                "question": item.get("question", ""),
                "supporting_facts": item.get("supporting_facts", []),
                "answer": item.get("answer", ""),
            }
            cleaned_data.append(cleaned_item)

        # Write cleaned data
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

        print(f"Processing completed! Processed {len(cleaned_data)} items")
        print(f"Cleaned data saved to: {output_file}")

        # Display processing statistics
        print(f"\nStatistics:")
        print(f"- Total items: {len(cleaned_data)}")
        questions_with_answers = sum(1 for item in cleaned_data if item["answer"])
        print(f"- Items with answers: {questions_with_answers}")
        print(f"- Items without answers: {len(cleaned_data) - questions_with_answers}")

    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error during processing: {e}")


def preview_data(file_path, num_samples=3):
    """
    Preview processed data

    Args:
        file_path (str): File path
        num_samples (int): Number of samples to preview
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"\nPreview of first {min(num_samples, len(data))} items:")
        print("=" * 50)

        for i, item in enumerate(data[:num_samples]):
            print(f"\nItem {i+1}:")
            print(f"Question: {item['question']}")
            print(f"Supporting facts: {item['supporting_facts']}")
            print(f"Answer: {item['answer']}")
            print("-" * 30)

    except Exception as e:
        print(f"Error during data preview: {e}")


if __name__ == "__main__":
    input_file = "2wikimultihopqa.json"
    output_file = "2wikimultihopqa_clean.json"

    clean_json_data(input_file, output_file)

    preview_data(output_file)
