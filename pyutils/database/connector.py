import os
import shutil


def manage_database(source_db_file: str, backup_db_file: str) -> None:
    """Manage a database file with interactive backup and restore options.

    Presents a menu to backup (copy source → backup) or restore (copy backup → source).
    """
    if not os.path.exists(source_db_file):
        print(f"Source database not found: {source_db_file}")
        return

    def backup_database() -> None:
        try:
            shutil.copy2(source_db_file, backup_db_file)
            print("Backup successful.")
        except OSError as e:
            print(f"Backup failed: {e}")

    def restore_database() -> None:
        try:
            shutil.copy2(backup_db_file, source_db_file)
            print("Restore successful.")
        except OSError as e:
            print(f"Restore failed: {e}")

    while True:
        print("Options:")
        print("1. Backup Database")
        print("2. Restore Database")
        print("3. Quit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == "1":
            backup_database()
        elif choice == "2":
            restore_database()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
