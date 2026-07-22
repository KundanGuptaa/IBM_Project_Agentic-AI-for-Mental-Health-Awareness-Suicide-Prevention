from dotenv import load_dotenv
load_dotenv(override=True)
import os, flask
flask.Flask.run = lambda *a, **k: None
import app as mg

print("=== MindGuard AI – Live Granite Test ===")
print(f"API Key    : {'[OK]' if mg.WATSONX_API_KEY != 'your-api-key-here' else '[MISSING]'}")
print(f"Project ID : {'[OK]' if mg.WATSONX_PROJECT_ID != 'your-project-id-here' else '[MISSING]'}")
print(f"URL        : {mg.WATSONX_URL}")
print()

ok = mg._init_granite_model()
print(f"Granite init : {'SUCCESS' if ok else 'FAILED (see error above)'}")
print(f"Model object : {type(mg.granite_model).__name__}")
print()

if ok:
    print("Sending test prompt to IBM Granite...")
    resp = mg.generate_response("In one sentence, what is mindfulness?", max_tokens=80)
    print(f"Granite response: {resp}")
else:
    print("Skipping live call – model not initialized.")
