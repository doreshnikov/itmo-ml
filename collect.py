import sys
import re


def read_lines(filename):
    file = open(filename, encoding='utf-8')
    lines = list(map(str.rstrip, file.readlines()))
    file.close()
    return lines


metainfo = dict()
filenames = []
trailing_code = []
state = 0
for line in read_lines('gen/main.pyt'):
    if len(line) == 0:
        state += 1
    if line.startswith('#'):
        assert state == 0
        key_value = line[1:].split('=', 1)
        metainfo[key_value[0]] = key_value[1] if len(key_value) > 1 else None
    elif line.startswith('@include '):
        filenames.append(line.split()[1])
    elif state == 2:
        trailing_code.append(line)

packages = list(map(lambda line: line.split('.')[0].replace('/', '.'), filenames))
package_choice = f'({"|".join(packages)})'
banned_imports = re.compile(f'(from {package_choice} import|import {package_choice})')

target = metainfo.get('target', 'gen/main.py')
sys.stdout = open(target, 'w', encoding='utf-8')
imports = []
lines = []
for it, filename in enumerate(filenames):
    package = packages[it]
    source_code = read_lines(filename)
    head = True
    for line in source_code:
        if re.match(banned_imports, line):
            continue
        elif line.startswith('import') or line.startswith('from'):
            imports.append(line)
        else:
            if 'main' in metainfo and \
                    metainfo['main'] != package and \
                    metainfo['main'] != '@all' and \
                    line == "if __name__ == '__main__'":
                break
            if len(line) > 0 or not head:
                lines.append(line)
                head = False
    lines.append('\n')

for line in imports + ['\n'] + lines + trailing_code:
    print(line)
sys.stdout.close()
