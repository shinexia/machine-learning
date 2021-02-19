#!/usr/bin/env python3


import argparse
import logging
import sys
import os
import subprocess
import re


"""给requirements.txt添加版本号
Usage: python versions.py -i requirements.txt
"""


def get_conda_deps() -> dict:
    all_deps_dict = {}
    out = subprocess.Popen(['conda', 'list'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, _ = out.communicate()
    stdout = stdout.decode('utf8')
    if 'ERROR' in stdout:
        logging.error('COMMAND FAILED: \n%s', stdout)
        sys.exit(1)
    for line in stdout.split('\n'):
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue
        spans = re.split('[ \t]+', line)
        if len(spans) < 3:
            logging.error('COMMAND FAILED:\nline:%s\nspans:(%d)[%s]\nstdout:\n%s', line, len(spans), ','.join(spans), stdout)
            sys.exit(1)
        name = spans[0]
        version = spans[1]
        all_deps_dict[name.lower()] = (name, version)
    return all_deps_dict


def get_pip_deps() -> dict:
    all_deps_dict = {}
    out = subprocess.Popen(['pip', 'freeze'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, _ = out.communicate()
    stdout = stdout.decode('utf8')
    if 'ERROR' in stdout:
        logging.error('COMMAND FAILED: \n%s', stdout)
        sys.exit(1)
    for line in stdout.split('\n'):
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue
        spans = re.split('==', line)
        if len(spans) != 2:
            continue
        name = spans[0]
        version = spans[1]
        all_deps_dict[name.lower()] = (name, version)
    return all_deps_dict


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str,
                        default='requirements.txt')
    parser.add_argument('--output', '-o', type=str,
                        default=None)
    parser.add_argument('--clean', default=False, action='store_true', dest='clean', help='clean versions')

    args = parser.parse_args()

    all_deps_dict = get_conda_deps()
    pip_deps_dict = get_pip_deps()
    for k, v in pip_deps_dict.items():
        if k not in all_deps_dict:
            all_deps_dict[k] = v
    logging.info('all deps count: %s', len(all_deps_dict))

    input_file = args.input
    logging.info('reading %s', input_file)

    results = []
    with open(args.input, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            spans = line.split('==')
            if len(spans) > 2:
                logging.error('invalid line: %s', line)
                sys.exit(1)
            name = spans[0].lower()
            name_prefix = re.sub('\[[^]]+\]', '', name)
            if len(name) > len(name_prefix) and name.startswith(name_prefix):
                name_suffix = name[len(name_prefix):]
            else:
                name_prefix = name
                name_suffix = ''
            dep = all_deps_dict.get(name, None) or all_deps_dict.get(name_prefix, None)
            if args.clean:
                if dep is None:
                    result = name
                else:
                    d = dep[0]
                    if len(name_suffix) > 0 and not d.endswith(name_suffix):
                        d = d + name_suffix
                    result = d
            else:
                if dep is None:
                    logging.error('dep for `%s` not found', name)
                    sys.exit(1)
                d = dep[0]
                if len(name_suffix) > 0 and not d.endswith(name_suffix):
                    d = d + name_suffix
                result = d + '==' + dep[1]
            results.append(result)

    output_file = args.output or args.input
    logging.info('writing to %s', output_file)
    with open(output_file, 'w+', encoding='utf8') as f:
        f.write('\n'.join(results))
        f.write('\n')


if __name__ == "__main__":
    main()
