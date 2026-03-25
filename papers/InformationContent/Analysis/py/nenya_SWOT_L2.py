""" Run Nenya on the SWOT L2 dataset """

from nenya import workflow
import info_defs

from IPython import embed

def main(task:str):
    dataset = 'SWOT_L2'
    pdict = info_defs.grab_paths(dataset)
    if task == 'train':
        workflow.train(pdict['opts_file'], load_epoch=None, debug=False)
    elif task == 'evaluate':
        workflow.evaluate(pdict['opts_file'], pdict['preproc_file'], local_model_path=pdict['path'],
                          latents_file=pdict['latents_file'], debug=False)
    elif task == 'chk_latents':
        workflow.chk_latents(dataset, pdict['latents_file'], pdict['preproc_file'], 100)
    else:
        raise ValueError(f"Unknown task: {task}")

# Command line execution
if __name__ == '__main__':
    task = workflow.return_task()
    main(task)
