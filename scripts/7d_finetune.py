import argparse
import src.utils_finetune_closedloop as utils_finetune

def execution(subj):
    print(f'Initializing for Finetune Transfer learning for closed loop experiments...')
    utils_finetune.run(subj)
    print('Finished')

def main():
    for subj in FLAGS.subject:
        print(subj)
        execution(subj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--subject", nargs='+', default=['X01'], help="Subject.")
    FLAGS, unparsed = parser.parse_known_args()
    main()