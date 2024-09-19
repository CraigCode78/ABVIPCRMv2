# test_openai.py

from openai.error import RateLimitError

def test_import():
    try:
        error = RateLimitError("Test error")
        print("Import successful:", error)
    except ModuleNotFoundError as e:
        print("Import failed:", e)

if __name__ == "__main__":
    test_import()