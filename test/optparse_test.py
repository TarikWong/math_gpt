# -*- coding: utf-8 -*-
# @Time : 2023/10/18 2:11 下午
# @Author : tuo.wang
# @Version : 
# @Function :
# 导入包
import optparse

if __name__ == '__main__':
    mdopt = optparse.OptionParser()
    mdopt.add_option('-t', '--template', dest='template', type='string', default='a', help='template file.')
    mdopt.add_option('-x', '--extensions', dest='extensions', type='string', default='', help='extensions.')
    mdopt.add_option('-o', '--output', dest='output', type='string', default='', help='output file.')
    mdopt.add_option('-d', '--deny', dest='deny', type='string', default='', help='deny file.')

    options, args = mdopt.parse_args()
    print('\nshow: {}  {}'.format(options, args))

    template = options.template,
    extensions = options.extensions.split(','),
    output = options.output,
    deny = options.deny.split(','),
    md = args

    print("\nshow_2: %s, %s, %s, %s, %s\n" % (template, extensions, output, deny, md))
    print("template 类型：", type(template))
    print("extensions 类型：", type(extensions))
