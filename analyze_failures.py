import re, json

with open('test_output/vector_matching/diagnostics.json') as f:
    content = f.read()

idx = content.find('// ---- FULL DATA ----')
json_str = content[idx:]
json_str = re.sub(r'//[^\n]*\n', '', json_str)
json_str = json_str.strip()
json_str = json_str.replace('"n/a,"', '"n/a","')
data = json.loads(json_str)

fails = []
for e in data:
    s = e.get('score', 0)
    iou = e.get('iou')
    if iou is None:
        iou = 0.0
    passed = s > 70.0 and iou > 0.90
    if not passed:
        fails.append(e)

print(f'Total entries: {len(data)}, Failures: {len(fails)}')
print()
for i, f2 in enumerate(fails):
    iou = f2.get('iou') or 0
    print(f"{i+1:2d}. {f2['bg']:45s} {f2['shape']:30s} score={f2['score']:6.1f}% iou={iou:.2f}  status={f2['status']}")

