import re

lines = open('.env', encoding='utf-8').readlines()
print(f'Total lines: {len(lines)}')
print()

EXPECTED_KEYS = {
    'WATSONX_API_KEY',
    'WATSONX_PROJECT_ID',
    'WATSONX_URL',
}

found_keys = set()

for i, line in enumerate(lines, 1):
    raw     = line.rstrip('\n')
    stripped = raw.strip()

    if not stripped or stripped.startswith('#'):
        print(f'  Line {i:2d}: [comment/blank]')
        continue

    if '=' not in stripped:
        print(f'  Line {i:2d}: [ERROR - no = found]  ->  {raw[:60]!r}')
        continue

    key, _, val = stripped.partition('=')
    key_raw = key
    key     = key.strip()
    val_raw = val
    val     = val.strip()

    found_keys.add(key)
    issues = []

    # Leading/trailing space on key
    if key_raw != key_raw.strip():
        issues.append('spaces around key name')

    # Leading/trailing space on raw value
    if val_raw != val_raw.strip():
        issues.append('leading/trailing spaces in value')

    # Strip surrounding quotes for value analysis
    inner = val
    if (inner.startswith('"') and inner.endswith('"')) or \
       (inner.startswith("'") and inner.endswith("'")):
        inner = inner[1:-1]

    # Empty
    if inner == '':
        issues.append('VALUE IS EMPTY')

    # Placeholder text
    if re.search(r'your[-_]', inner, re.I) or 'placeholder' in inner.lower():
        issues.append('still a placeholder - replace with real value')

    # Unquoted spaces
    if ' ' in val and not (val.startswith('"') or val.startswith("'")):
        issues.append('value contains spaces but is not quoted')

    # Mismatched quotes
    if val.startswith('"') and not val.endswith('"'):
        issues.append('unclosed double-quote')
    if val.startswith("'") and not val.endswith("'"):
        issues.append('unclosed single-quote')

    # Trailing carriage return (Windows \r\n)
    if line.endswith('\r\n'):
        pass  # fine, dotenv handles it
    if '\r' in inner:
        issues.append('carriage return (\\r) inside value')

    # Mask value for display
    if len(inner) > 8:
        masked = inner[:3] + '*' * (len(inner) - 5) + inner[-2:]
    elif inner:
        masked = '***'
    else:
        masked = '[empty]'

    status = 'OK' if not issues else 'WARN: ' + ' | '.join(issues)
    print(f'  Line {i:2d}: [{status}]')
    print(f'           Key   = {key!r}')
    print(f'           Value = {masked}  (length={len(inner)})')
    print()

# Check for missing expected keys
missing = EXPECTED_KEYS - found_keys
if missing:
    print('MISSING REQUIRED KEYS:')
    for k in sorted(missing):
        print(f'  - {k}')
else:
    print('All 3 required keys are present.')

# Check dotenv actually loads them
print()
print('--- dotenv load test ---')
from dotenv import load_dotenv
import os
load_dotenv(override=True)
for k in sorted(EXPECTED_KEYS):
    v = os.getenv(k, '')
    if len(v) > 8:
        masked = v[:3] + '*' * (len(v) - 5) + v[-2:]
    elif v:
        masked = '***'
    else:
        masked = '[NOT LOADED]'
    print(f'  os.getenv({k!r}) = {masked}  (len={len(v)})')
