def clean_string(string):
    bad_chars = [';', ':', '!', "*", ",", "#", "?", "-", "(", ")", "[", "]", "{", "}", "/", "\\"]
    for c in bad_chars:
        string = string.replace(c, '')
    return string