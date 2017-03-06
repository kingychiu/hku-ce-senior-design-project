class FileIO:
    @staticmethod
    def read_first_lines(num, file_path):
        with open(file_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()
            f.close()
            return lines[:num]

    @staticmethod
    def num_lines(file_path):
        with open(file_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()
            f.close()
            return len(lines)

    @staticmethod
    def read_file_to_lines(file_path):
        with open(file_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()
            f.close()
            return lines

    @staticmethod
    def write_lines_to_file(file_path, lines):
        with open(file_path, 'w', encoding='utf8') as f:
            f.write('\n'.join(lines))
