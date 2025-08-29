from hirag_prod.schema import (
    File,
    Chunk,
)

# test to dict

def test_file_to_dict():
    file = File(
        documentKey="1",
        text="This is a test file.",
        fileName="test.md",
        uri="http://example.com/test.md",
        private=False,
        knowledgeBaseId="kb1",
        workspaceId="ws1",
    )
    print(file.id)

if __name__ == "__main__":
    test_file_to_dict()