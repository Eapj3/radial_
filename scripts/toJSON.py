import json
import os
import sys


# Transforms NASA .tbl to JSON format
# $python3 example.tbl /path/to/data/folder
# Creates a new directory named as the STAR_ID and saves the data from example.tbl into JSON
# file named as the INSTRUMENT


def restruct(lines: list[str], headers: list = None, extra: list = None,
             first: list = None, second: list = None, third: list = None, metadata: dict = None) -> dict:
    head = 0
    if headers is None:
        headers = []
    if extra is None:
        extra = []
    if first is None:
        first = []
    if second is None:
        second = []
    if third is None:
        third = []
    if metadata is None:
        metadata = {}

    for line in lines:
        if '\\' in line:
            vals = line[line.index('\\') + 1:].strip().split('=')
            if len(vals) == 2:
                pass
            else:
                vals = [x for x in vals if x.strip() != '']

            if len(vals) > 0:
                try:
                    metadata[vals[0].strip()] = vals[1].strip()
                except IndexError:
                    metadata[vals[0].strip()] = None

        elif '|' in line:
            if head == 0:
                head += 1
                vals = line[line.index('|') + 1:].strip().split('|')
                vals = [x.strip() for x in vals if x.strip() != '']
                headers = vals
            else:
                extra.append(line)

        else:
            values = line.split()
            first.append(values[0])
            second.append(values[1])
            third.append(values[2])

    metadata['extra_headers'] = extra
    data_dict = {headers[0]: first, headers[1]: second, headers[2]: third, 'metadata': metadata}

    return data_dict


def main(path: str, out_prepend: str) -> None:
    with open(path) as da:
        ll = da.readlines()
    data_f = restruct(ll)

    out_folder = data_f['metadata']['STAR_ID'].strip('"').strip("'").replace(' ', '_')

    if '/' in out_prepend:
        out_prepend += '/'
        out_folder += '/'
    else:
        out_prepend += '\\'
        out_folder += '\\'

    instrument = data_f['metadata']['INSTRUMENT'].strip('"').strip("'").replace(' ', '_')
    if 'spectrograph' in instrument:
        instrument = 'spectrograph'

    try:
        os.mkdir(out_prepend + out_folder[:-1])
    except FileExistsError:
        pass

    while True:
        try:
            with open(out_prepend + out_folder + instrument + '.json', 'x') as outfile:
                json.dump(data_f, outfile, indent=4)
            break
        except FileExistsError:
            instrument += 'V'
            print(out_folder)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
