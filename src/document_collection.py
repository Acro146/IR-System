import os
from dataclasses import dataclass, field
from pathlib import Path
import re


@dataclass(frozen=True, order=True)
class Document:
    document_id: int
    document_path: Path = field(compare=False)


@dataclass
class DocumentCollection:
    directory: Path

    def __iter__(self):
        files = os.scandir(self.directory)
        txt_files = []
        for file in sorted(files, key=lambda x: self.natural_keys(x.name)):
            if self.is_text_file(file):
                txt_files.append(file)

        files = txt_files
        for i, file in enumerate(files):
            if not file.is_file():
                continue
            yield Document(i, Path(file.path))

    @staticmethod
    def is_text_file(file):
        return file.is_file() and file.name.endswith(".txt")

    @staticmethod
    def atof(text):
        try:
            retval = float(text)
        except ValueError:
            retval = text
        return retval

    def natural_keys(self, text):
        return [self.atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]
