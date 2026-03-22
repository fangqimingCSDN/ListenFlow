# -*- coding: utf-8 -*-
import pathlib

base = pathlib.Path('d:/B-Work/PyCharm/2025/speech_proj/scripts')

secs = [
    '_arch_sec0.txt',
    '_arch_sec1.txt',
    '_arch_sec2.txt',
    '_arch_sec3.txt',
    '_arch_sec4.txt',
    '_arch_sec5.txt',
    '_arch_sec6a.txt',
    '_arch_sec6b.txt',
    '_arch_sec7.txt',
]

parts = []
for s in secs:
    p = base / s
    parts.append(p.read_text(encoding='utf-8'))
    print(f'{s}: {len(parts[-1])} chars')

final = '\n'.join(parts)
out = pathlib.Path('d:/B-Work/PyCharm/2025/speech_proj/ARCHITECTURE.md')
out.write_text(final, encoding='utf-8')
lines = final.count('\n') + 1
print(f'\nFinal ARCHITECTURE.md: {len(final)} chars, {lines} lines')
