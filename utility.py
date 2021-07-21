def clean_string(string):
    bad_chars = [';', ':', '!', '*', ',', '#', '?', '-', '(', ')', '[', ']', '{', '}', '/', '\\', '"', "'"]
    for c in bad_chars:
        string = string.replace(c, '')
    return string

def remove_duplicate_lines(input_file, output_file):
    lines_seen = set()
    outfile = open(output_file, 'w')
    for line in open(input_file, 'r'):
        if line not in lines_seen:
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()