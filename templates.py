import copy
import itertools

import util

UNIQUE_TEMPLATES = {
    'beginning': {'child': 'child: "{}"',
                  'parent': 'parent: "{}"'},
    # 'foo': {'child': 'foo: "{}"',
    #         'parent': 'bar: "{}"'},
    'pro/con': {'pro': 'pro: "{}"',
                'con': 'contra: "{}"'},
    'combined': {'child': 'child: "{}" parent: "{}"'},
    'combined_short': {'child': '"{}" parent: "{}"'},
    'combined_pro/con': {'pro': 'pro: "{}" parent: "{}"',
                         'con': 'contra: "{}" parent: "{}"'},
    'all': {},
    'all-meaningless': {},
    'all-all': {}
}

MEANINGLESS_UNIQUE_TEMPLATES = {
    # 'foo': {'child': 'foo: "{}"',
    #         'parent': 'bar: "{}"'},
    'beginning': {'child': 'foo: "{}"',
                  'parent': 'bar: "{}"'},
    'pro/con': {'pro': 'baz: "{}"',
                'con': 'qux: "{}"'},
    'combined': {'child': 'foo: "{}" bar: "{}"'},
    'combined_short': {'child': '"{}" bar: "{}"'},
    'combined_pro/con': {'pro': 'baz: "{}" bar: "{}"',
                         'con': 'qux: "{}" bar: "{}"'},
    'all': {},
    'all-meaningless': {},
    'all-all': {}
}

POSSIBLE_TEMPLATES = {
    'parent': ['parent'],
    'pro': ['child', 'pro'],
    'con': ['child', 'con']
}

standard_template = 'beginning'
TEMPLATES = copy.deepcopy(UNIQUE_TEMPLATES)
for k, v in TEMPLATES.items():
    if 'child' not in v:
        v['child'] = TEMPLATES[standard_template]['child']
    if 'parent' not in v:
        v['parent'] = TEMPLATES[standard_template]['parent']


current = {}


def init():
    combine = UNIQUE_TEMPLATES | {k + '_meaningless': v for k, v in MEANINGLESS_UNIQUE_TEMPLATES.items()}
    current['templates'], current['unique_templates'] = ((MEANINGLESS_UNIQUE_TEMPLATES, MEANINGLESS_UNIQUE_TEMPLATES)
                                                         if util.args['template_id'] in ['foo', 'all-meaningless'] else
                                                         ((combine, combine) if util.args['template_id'] == 'all-all'
                                                          else (TEMPLATES, UNIQUE_TEMPLATES)))
    current['template_id'] = util.args['template_id'] if util.args['template_id'] != 'foo' else 'beginning'


# primary template is used as main template for eval
def format_primary(text: str, node_type: str, use_templates: bool):
    if not use_templates:
        return text
    return current['unique_templates'][standard_template][node_type].format(text)


def format_all_possible(text: str, parent_text: str, node_type: str, use_templates: bool, parent=True):
    if not use_templates:
        return [text]

    if not parent:
        text = f'not {text}'

    if node_type == 'parent':
        # use one parent representation to avoid repetition of same parent in the same batch
        return [format_primary(text, node_type, use_templates)]

    if util.args['template_id'].startswith('all'):
        return list(itertools.chain.from_iterable(
            format_using_template(
                text, parent_text, node_type, t,
                current['unique_templates'])
            for t in current['unique_templates'].keys()))
    else:
        return format_using_template(text, parent_text, node_type, current['template_id'], current['templates'])


def format_using_template(text: str, parent_text: str, node_type: str, template_id: str, templates: dict):
    return [templates[template_id][t].format(*([text, parent_text] if 'combined' in template_id else [text]))
            for t in POSSIBLE_TEMPLATES[node_type] if t in templates[template_id]]


# util.args = {}
# util.args['template_id'] = 'all'
# util.args['template_id'] = 'all-meaningless'
# util.args['template_id'] = 'foo'
# util.args['template_id'] = 'all-all'
# init()
#
# print('child')
# print(*[x for x in format_all_possible('child text', 'parent text', 'pro', True, True)], sep='\n')
# print('parent')
# print(*[x for x in format_all_possible('parent text', 'grand text', 'parent', True, True)], sep='\n')
