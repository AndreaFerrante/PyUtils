import sqlite3
import shutil


def manage_database(source_db_file:str, backup_db_file:str):

    """
    Manage a database by providing options to backup and restore.

    This function encapsulates two nested functions for backing up and restoring
    a database. It presents the user with a simple menu to choose between these
    options or to quit the program.

    Parameters:
    - source_db_file (str): The path to the source database file that needs to be managed.
    - backup_db_file (str): The path where the backup of the source database will be stored.

    Nested Functions:
    - backup_database(): Attempts to create a backup of the source database.
    - restore_database(): Attempts to restore the source database from its backup.

    Both nested functions handle exceptions and will print out whether the operation
    was successful or the reason for any failure.

    Menu Options:
    1. Backup Database: Executes backup_database().
    2. Restore Database: Executes restore_database().
    3. Quit: Exits the while loop, effectively terminating this function's execution.

    Example:
    >>> manage_database("source.db", "backup.db")
    Options:
    1. Backup Database
    2. Restore Database
    3. Quit
    Enter your choice (1/2/3): 1
    Backup successful.
    """

    def backup_database():
        try:
            shutil.copy2(source_db_file, backup_db_file)
            print("Backup successful.")
        except Exception as e:
            print(f"Backup failed: {str(e)}")

    def restore_database():
        try:
            shutil.copy2(backup_db_file, source_db_file)
            print("Restore successful.")
        except Exception as e:
            print(f"Restore failed: {str(e)}")

    while True:
        print("Options:")
        print("1. Backup Database")
        print("2. Restore Database")
        print("3. Quit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            backup_database()
        elif choice == '2':
            restore_database()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")