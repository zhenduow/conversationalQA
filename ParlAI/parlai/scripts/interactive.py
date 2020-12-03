#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic script which allows local human keyboard input to talk to a trained model.

Examples
--------

.. code-block:: shell

  python examples/interactive.py -m drqa -mf "models:drqa/squad/model"

When prompted, enter something like: ``Bob is Blue.\\nWhat is Bob?``

Input is often model or task specific, but in drqa, it is always
``context '\\n' question``.
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.scripts.script import ParlaiScript
from parlai.agents.local_human.local_human import LocalHumanAgent

import random


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Interactive chat with a model')
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument(
        '--display-prettify',
        type='bool',
        default=False,
        help='Set to use a prettytable when displaying '
        'examples with text candidates',
    )
    parser.add_argument(
        '--display-ignore-fields',
        type=str,
        default='label_candidates,text_candidates',
        help='Do not display these fields',
    )
    parser.add_argument(
        '-it',
        '--interactive-task',
        type='bool',
        default=True,
        help='Create interactive version of task',
    )
    '''
    parser.add_argument(
        '-hi',
        '--history',
        type=str,
        default='',
        help='context history beyond the initial query',
    )
    parser.add_argument(
        '-hist',
        '--human_input',
        type=str,
        default='Hello.',
        help='Human input message to the agent.',
    )
    parser.add_argument(
        '-sampcans',
        '--sampled_candidates',
        type=str,
        default='',
        help='Sampled candidates',
    )
    '''
    parser.set_defaults(interactive_mode=True, task='interactive')
    LocalHumanAgent.add_cmdline_args(parser)
    return parser


def create_reranker(opt, print_parser=None):
    if print_parser is not None:
        if print_parser is True and isinstance(opt, ParlaiParser):
            print_parser = opt
        elif print_parser is False:
            print_parser = None
    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: interactive should be passed opt not Parser ]')
        opt = opt.parse_args()

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()
    human_agent = LocalHumanAgent(opt)
    world = create_task(opt, [human_agent, agent])
    return world

    # Show some example dialogs:
    #while not world.epoch_done():
        # modify world.parley to take sentence as input
        # world.acts[1]['text_candidate'] returns the rank

def rerank(world, human_input, history, sampled_candidates):
    try:
        print(world.acts[1]['text_candidates'])
    except:
        pass
    #world.parley(human_input, history, sampled_candidates)
    #return world.acts[1]['text_candidates'] 
    return world.parley(human_input, history, sampled_candidates)
        #if opt.get('display_examples'):
        #    print("---")
        #    print(world.display())
    


class Interactive(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return create_reranker(self.opt, print_parser=False)


if __name__ == '__main__':
    random.seed(42)
    Interactive.main()
