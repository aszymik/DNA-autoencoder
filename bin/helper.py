import os
import shutil
from .common import get_seq_id

def rewrite_fasta(file, outdir=None, name_pos=None, force=False):
    # check if there is more than one sequence in the given file
    with open(file, 'r') as f:
        i = 0
        line = f.readline()
        while i < 2 and line:
            if line.startswith('>'):
                i += 1
            line = f.readline()
        if i == 1:
            print('No rewriting was done: given file contains only one sequence.')
            return 1, file
    if outdir is None:
        outdir, name = os.path.split(file)
        namespace, _ = os.path.splitext(name)
        outdir = os.path.join(outdir, namespace)
    if os.path.isdir(outdir):
        num_files = len([el for el in os.listdir(outdir) if el.endswith('.fasta')])
    if force or not os.path.isdir(outdir):
        if os.path.isdir(outdir):
            shutil.rmtree(outdir)
            print('Forced to remove directory {} with {} fasta files'.format(outdir, num_files))
        os.mkdir(outdir)
        num_files = 0
    if num_files == 0:
        i = 0
        sequence_written = True
        with open(file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    assert sequence_written, 'No sequence written for {}'.format(filename)
                    sequence_written = False
                    filename = get_seq_id(line, name_pos) + '.fasta'
                    w = open(os.path.join(outdir, filename), 'w')
                    w.write(line)
                    i += 1
                else:
                    w.write(line)
                    w.close()
                    sequence_written = True
        assert sequence_written, 'No sequence written for {}'.format(filename)
        print('Based on {} {} sequences were written into separated files in {}'.format(file, i, outdir))
        return i, outdir
    else:
        print('Directory {} with {} fasta files already exists - no rewritting was done.'.format(outdir, num_files))
        return num_files, outdir