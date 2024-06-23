from pprint import pprint

import matplotlib.font_manager as fm


def main() -> None:
    font_set = {
        fm.FontProperties(fname=font).get_name()
        for font in fm.findSystemFonts()
    }
    pprint(font_set)


if __name__ == "__main__":
    main()
