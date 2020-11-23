"""
    Add all the directories in the project to sys.path.

    Author: Akshay Seshadri
"""

from pathlib import Path
import sys

def find_subdir_recursive(base = '.', maxdepth = 2):
    """
        Finds all the subdirectories of the base directory recursively, until maxdepth level is reached from the base directory.

        For example, base = '.' and maxdepth = 2 gives the current directory (level = 0), all subdirectories of current directory (level = 1),
        and the subdirectories of these subdirectories (level = 2).

        Symlinks are discarded.

        Arguments:
            - base     : path to the directory where the search begins
            - maxdepth : number of levels from the base directory to search for subdirectories

        Returns:
            - a list of Path objects of the subdirectories of the base directory of maxdepth depth.
              The base directory is always included as the first element.
    """
    # ensure that base directory exists
    base_dir = Path(base)
    if not base_dir.exists():
        raise ValueError("base: no such directory exists")

    # list of subdirectories
    subdir_list = [base_dir]
    # recursively find the subdirectories until maxdepth is reached
    for level in range(1, int(maxdepth) + 1):
        # constuct a pattern that searches subdirectories 'level' depth of the current directory
        search_pattern = '/'.join(['*'] * level)

        # get all the files in the directories (including dirnames) given the search path
        # Path doesn't support finding "only" directories; we use it because it's convenient and portable
        all_files_level = base_dir.glob(search_pattern)

        # extract only the directors from the globbed files
        # is_dir returns True for symlinks, so they need to be specifically discarded
        all_subdirs_level = [path for path in all_files_level if path.is_dir() and not path.is_symlink()]

        subdir_list.extend(all_subdirs_level)

    return subdir_list

def add_project_path_sys(root, depth, blacklist = ['__pycache__']):
    """
        Given the root directory for the project and the depth (number of levels of subdirectories) of the project,
        adds all subdirectories to the end of sys.path.

        The directories in blacklist are not added to sys.path. Note that the string given in blacklist does a greedy match.
        For example, if two different directorys a1/b and a2/b share the same name 'b', supplying 'b' will blacklist them both.
        If specifically a1/b needs to be blacklisted, supply 'a1/b'.
        Note that the blacklist is compared with the paths relative to the root directory, and not the absolute paths.
        This is done to avoid matching anything with parent directories of the root directory, which are not of any concern.
        Do not provide an absolute path in blacklist.
        
        Arguments:
            - root      : path to the root directory of the project
            - depth     : number of levels of subdirectories in the project
            - blacklist : a list of subdirectories that should not be added to sys.path;
                          paths should be relative to the root directory; absolute paths should not be supplied
    """
    # absolute path of the root directory
    root_dir = Path(root).resolve()
    if not root_dir.exists():
        raise ValueError("root: no such directory exists")
    if not root_dir.is_dir() or root_dir.is_symlink():
        raise ValueError("root: must be a path to a directory; symlinks are not allowed")

    # get all subdirectories of the root directory
    root_subdirs = find_subdir_recursive(base = root_dir, maxdepth = depth)

    # get the abolsute paths of each subdirectory
    root_subdirs = [path.resolve() for path in root_subdirs]

    # ensure that blacklist is a list of strings
    blacklist = [str(blk_path) for blk_path in blacklist]
    # remove './' from the beggining of the blacklisted paths, if it exists
    blacklist = [blk_path.lstrip('./') if blk_path.startswith('./') else blk_path for blk_path in blacklist]

    # remove all blacklisted directories from the computed subdirectories of the root directory
    # ensure that we are comparing elements of blacklist against the relative path with the root directory
    # we convert the Path objects to plain strings because we need to add it to sys.path
    root_subdirs = [str(path) for path in root_subdirs if not any([blk_path in str(path.relative_to(root_dir)) for blk_path in blacklist])]

    # get the new paths that are not already in sys.path
    root_dir_new = [path for path in root_subdirs if not path in sys.path]

    # add the whitelisted (and new) abolsute paths to the end of sys.path
    sys.path.extend(root_subdirs)

### add current project dirs to syspath: these statements will be run on an import
# we are one level below root folder
level = 1
# project path for the current project
project_root_dir = Path(__file__).parents[level]
# the current project has a depth of 2, add all the project dirs recursively
add_project_path_sys(root = project_root_dir, depth = 2, blacklist = ['__pycache__', '.ipynb_checkpoints/', 'yaml_files', 'estimator_files', 'outcome_files'])
