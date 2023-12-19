import os
import re
import pkg_resources

def find_imported_libraries(directory):
    imported_libraries = {}

    file_extensions = ['.py']  

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(tuple(file_extensions)):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    imports = re.findall(r'^\s*import\s+([^\s]+)', content, re.MULTILINE)
                    imports += re.findall(r'^\s*from\s+([^\s]+)', content, re.MULTILINE)
                    for library in imports:
                        try:
                            version = pkg_resources.get_distribution(library).version
                            imported_libraries[library] = version
                        except pkg_resources.DistributionNotFound:
                            imported_libraries[library] = 'unknown'

    return imported_libraries


current_directory = 'C://Users//Toktam//Documents//GitHub//BrainMRIAnalysis//'  
libraries2 = find_imported_libraries(current_directory)
for library, version in libraries2.items():
    print(f"{library}=={version}")
requirements_file = current_directory+"requirements.txt"
with open(requirements_file, 'w') as f:
    for library, version in libraries2.items():
        if version != "unknown":
            f.write(f"{library}=={version}\n")
        else:
            f.write(f"{library}\n")