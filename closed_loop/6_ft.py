import argparse
import utils_finetune_closedloop as utils_finetune
from pathlib import Path

def execution(test_subject, session):
    print(f'Initializing for Finetune Transfer learning for closed loop experiments...')
    for model_type in ['best_pretrain', 'best_finetune']:
        train_trial = [[1, 2], [0, 1], [0,2]]
        val_trials = [0,2,1]
        # here, best_finetune is the most recent fine_tuned model, thus for session 2 this is FTsession1
        session_num = int(session[-1])

        # model path and subjpath for future sessions
        model_path = f"closed_loop/final_models/{model_type}\{session}/EEGNET_{test_subject}"
        savepath_newmodel = f"closed_loop/final_models/models_for_closedloop/EEGNET_{test_subject}_{model_type}_{session}"

        # data path below
        testsubj_path = Path(f'./closed_loop\intermediate_files/{session}/{test_subject}_deep.pkl')
 
        print(f'Getting {model_type} for {test_subject}..')
        for i in range(3):

            train_trials = train_trial[i]#[1, 2]
            val_trial = val_trials[i] #0
            config={
            'batch_size' : 256,
            'epochs': 20,
            'receptive_field': 64, 
            'mean_pool':  8,
            'activation_type':  'elu',
            'network' : 'EEGNET',
            'model_type': model_type,
            'model_path': model_path,
            'savepath_newmodel': savepath_newmodel,
            'test_subj_path' : testsubj_path,
            'test_subject': test_subject,
            'train_trials': train_trials,
            'val_trial': val_trial,
            'CLsession': session_num,
            'ablation': 'all',
            'seed':  42,    
            'learning_rate': 0.001,
            'filter_sizing':  8,
            'D':  2,
            'dropout': 0.25}
            utils_finetune.train(config)
    print('Finished')

def main():
    for subj in FLAGS.subject:
        print(subj)
        execution(subj, FLAGS.session)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--subject", nargs='+', default=['X01'], help="Subject.")
    parser.add_argument("--session", type=str, default='session3', help="Subject.")
    FLAGS, unparsed = parser.parse_known_args()
    main()