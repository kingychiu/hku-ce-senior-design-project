
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