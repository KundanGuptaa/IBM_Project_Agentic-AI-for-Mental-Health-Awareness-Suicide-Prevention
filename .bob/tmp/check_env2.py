from dotenv import load_dotenv
import os

load_dotenv(override=True)

key  = os.getenv('WATSONX_API_KEY', '')
proj = os.getenv('WATSONX_PROJECT_ID', '')
url  = os.getenv('WATSONX_URL', '')

print('Key loaded    :', bool(key),  '| len:', len(key))
print('Proj loaded   :', bool(proj), '| len:', len(proj))
print('URL loaded    :', bool(url),  '| len:', len(url))
print()

for name, val in [('WATSONX_API_KEY', key), ('WATSONX_PROJECT_ID', proj), ('WATSONX_URL', url)]:
    leading_ws   = val != val.lstrip()
    trailing_ws  = val != val.rstrip()
    q1 = chr(34)  # double-quote
    q2 = chr(39)  # single-quote
    has_quotes   = val.startswith(q1) or val.startswith(q2) or val.endswith(q1) or val.endswith(q2)
    non_ascii    = any(ord(c) > 127 for c in val)
    print(f'{name}:')
    print(f'  leading whitespace  : {leading_ws}')
    print(f'  trailing whitespace : {trailing_ws}')
    print(f'  starts/ends w quote : {has_quotes}')
    print(f'  non-ASCII chars     : {non_ascii}')
    print()

print('All 3 values loaded successfully via dotenv.')
