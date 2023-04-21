import argparse
#generates the FASTA files (strings of amino acids that describe the entire protein sequence input to generate structure)

def generate_sequence_from_linker(linker):
    #amyloid beta 42 sequence, can change this if another amyloid protein is of interest
    abeta_42 = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'
    
    #linker sequence
    link = linker

    #this .pdb reference contains a pentamer of abeta42, therefore there will be 4 linkers and 5 main chain sequences
    n_mer = 5

    main_chain = ''
    for n in (n_mer-1):
        main_chain += abeta_42 + link

    main_chain += abeta_42
    
    return main_chain

def generate_fasta_file(linker):
    #generate the sequence
    sequence = generate_sequence_from_linker(linker)
    
    #create the fasta file content
    fasta_content = ">{}\n{}\n".format(linker, sequence)
    
    filename = linker + '.fasta'

    #write the fasta file
    with open(filename, "w") as fasta_file:
        fasta_file.write('fastas/' + fasta_content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--linker', help='Linker sequence for generating the fasta file')
    args = parser.parse_args

    if args.linker:
        generate_fasta_file(args.linker)
    else:
        print("Need to input a linker with --linker flag")