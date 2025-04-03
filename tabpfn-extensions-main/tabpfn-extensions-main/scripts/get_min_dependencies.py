from __future__ import annotations

import re


def main() -> None:
    """Extract minimum version constraints from dependencies in pyproject.toml.

    Creates a requirements.txt file with the minimum required versions
    of each dependency, converting lower bounds to exact version requirements.
    """
    with open("pyproject.toml") as f:
        content = f.read()

    # Find dependencies section using regex
    deps_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if deps_match:
        deps = [
            d.strip(" \"'")
            for d in deps_match.group(1).strip().split("\n")
            if d.strip()
        ]
        min_reqs = []
        for dep in deps:
            match = re.match(r'([^>=<\s]+)\s*>=\s*([^,\s"\']+)', dep)
            if match:
                package, min_ver = match.groups()
                min_reqs.append(f"{package}=={min_ver}")

        with open("requirements.txt", "w") as f:
            f.write("\n".join(min_reqs))


if __name__ == "__main__":
    main()
