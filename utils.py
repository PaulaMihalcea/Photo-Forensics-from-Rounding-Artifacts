def get_filename(file_path):

    tmp_filename = file_path.split('/')[-1]
    extension = tmp_filename.split('.')[-1]
    tmp_filename = tmp_filename.split('.')[:-1]

    if len(tmp_filename) > 1:
        filename = tmp_filename[0]
        for el in tmp_filename[1:]:
            filename += '.' + el
    else:
        filename = tmp_filename[0]

    return filename, extension
