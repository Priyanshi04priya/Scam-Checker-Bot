"""Make the password hash for ``ADMIN_PASSWORD_HASH``: ``python -m scambot.api.admin.hashpw``."""

import getpass

from argon2 import PasswordHasher


def main() -> int:
    password = getpass.getpass("Admin password: ")
    if password != getpass.getpass("Repeat: "):
        print("The passwords do not match.")  # noqa: T201
        return 1
    print(PasswordHasher().hash(password))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
