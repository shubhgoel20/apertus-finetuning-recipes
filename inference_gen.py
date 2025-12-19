sb_file_path = '/users/ddixit/scratch/apertus-project/apertus-finetuning-recipes/query_finetuned/submit_test_finetuned'
py_file_path = '/users/ddixit/scratch/apertus-project/apertus-finetuning-recipes/query_finetuned/test_finetuned'
ref_sb_file_path = '/users/ddixit/scratch/apertus-project/apertus-finetuning-recipes/query_finetuned/submit_test_finetuned.sbatch'
ref_py_file_path = '/users/ddixit/scratch/apertus-project/apertus-finetuning-recipes/query_finetuned/test_finetuned.py'


if __name__ == "__main__":

        with open(ref_sb_file_path, 'r') as file:
                sb_file_data = file.read()

        with open(ref_py_file_path, 'r') as file:
                py_file_data = file.read()

        for i in range(12, 36):
                my_sb_file_path = sb_file_path + f'_{i}.sbatch'
                my_py_file_path = py_file_path + f'_{i}.py'

                my_sbfile_data = sb_file_data.replace('_0', f'_{i}')
                my_sbfile_data = my_sbfile_data.replace('test_finetuned.py', f'test_finetuned_{i}.py')
                my_pyfile_data = py_file_data.replace('_0', f'_{i}')
                set = i // 6
                if set%2==0:
                        my_chk = 48
                        other_chk = 72
                else:
                        my_chk = 72
                        other_chk = 48
                my_pyfile_data = my_pyfile_data.replace(f'checkpoint-{other_chk}', f'checkpoint-{my_chk}')
                with open(my_sb_file_path, 'w') as file:
                        file.write(my_sbfile_data)
                with open(my_py_file_path, 'w') as file:
                        file.write(my_pyfile_data)