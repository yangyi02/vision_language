# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:25:55 2015

@author: yangyi05
"""

import io
import json
import argparse
from caption.pycocoevalcap.bleu.bleu import Bleu
from caption.pycocoevalcap.cider.cider import Cider
from caption.pycocoevalcap.meteor.meteor import Meteor
from caption.pycocoevalcap.rouge.rouge import Rouge
from caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from VQA.PythonHelperTools.vqaTools.vqa import VQA
from VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval


def evaluate_cap(result_json, gt_json, score_json):
    """
    Evaluate benchmark scores for image captioning
    """
    dataset = json.load(io.open(gt_json, 'rb'))
    imgToAnns_gts = {ann['image_id']: [] for ann in dataset}
    for ann in dataset:
        imgToAnns_gts[ann['image_id']] += [ann]

    result = json.load(io.open(result_json, 'rb'))
    imgToAnns_res = {ann['image_id']: [] for ann in result}
    for ann in result:
        imgToAnns_res[ann['image_id']] += [ann]

    imgIds = [ann['image_id'] for ann in result]

    gts, res = {}, {}
    for imgId in imgIds:
        gts[imgId] = imgToAnns_gts[imgId]
        res[imgId] = imgToAnns_res[imgId]

    evaluate_cap_score(res, gts, score_json)


def evaluate_coco_cap(resFile, annFile, score_json):
    """
    Evaluate benchmark scores for image captioning with microsoft coco style
    """
    # create coco object and cocoRes object
    dataset = json.load(io.open(annFile, 'rb'))
    imgToAnns_gts = {ann['image_id']: [] for ann in dataset['annotations']}
    for ann in dataset['annotations']:
        imgToAnns_gts[ann['image_id']] += [ann]

    result = json.load(io.open(resFile, 'rb'))
    imgToAnns_res = {ann['image_id']: [] for ann in result}
    for ann in result:
        imgToAnns_res[ann['image_id']] += [ann]

    imgIds = [ann['image_id'] for ann in result]

    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = imgToAnns_gts[imgId]
        res[imgId] = imgToAnns_res[imgId]

    evaluate_cap_score(res, gts, score_json)


def evaluate_cap_score(res, gts, score_json):
    # =================================================
    # Set up scorers
    # =================================================
    print 'tokenization...'
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # =================================================
    # Set up scorers
    # =================================================
    print 'setting up scorers...'
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    # =================================================
    # Compute scores
    # =================================================
    eval_score = {}
    for scorer, method in scorers:
        print 'computing %s score...' % (scorer.method())
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                eval_score[m] = sc
                print "%s: %0.3f" % (m, sc)
        else:
            eval_score[method] = score
            print "%s: %0.3f" % (method, score)

    # print output evaluation scores
    for metric, score in eval_score.items():
        print '%s: %.3f' % (metric, score)
    json.dump(eval_score, io.open(score_json, 'wb'))


def evaluate_qa(result_json, gt_json, score_json):
    """
    Evaluate benchmarks for image question answering.
    """
    question_answer = json.load(io.open(gt_json))

    dataset = json.load(io.open(gt_json, 'rb'))
    quesToAnns_gts = {ann['question_id']: [] for ann in dataset}
    for ann in dataset:
        quesToAnns_gts[ann['question_id']] += [ann]

    result = json.load(io.open(result_json, 'rb'))
    quesToAnns_res = {ann['question_id']: [] for ann in result}
    for ann in result:
        quesToAnns_res[ann['question_id']] += [ann]

    quesIds = [ann['question_id'] for ann in result]

    gts, res = {}, {}
    for quesId in quesIds:
        gts[quesId] = quesToAnns_gts[quesId]
        res[quesId] = quesToAnns_res[quesId]

    evaluate_qa_score(res, gts, quesIds, score_json)


def evaluate_qa_score(res, gts, quesIds, score_json):
    """
    Compute question anwering accuracy
    """
    accQA = []
    for quesId in quesIds:
        resAns = res[quesId][0]['answer'].replace('\n', ' ').replace('\t', ' ').strip()
        gtAcc = []
        for gtAnsDatum in gts[quesId]:
            acc = 1 if resAns == gtAnsDatum['answer'] else 0
            gtAcc.append(acc)
        avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
        accQA.append(avgGTAcc)
    eval_score = float(sum(accQA)) / len(accQA)
    print "QA accuracy: %0.3f" % eval_score
    json.dump(eval_score, io.open(score_json, 'wb'))


def evaluate_coco_vqa(resFile, quesFile, annFile, score_json):
    """
    Evaluate benchmarks for image question answering with microsoft coco VQA style
    """
    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)
    # create vqaEval object by taking vqa and vqaRes
    # n is precision of accuracy (number of places after decimal), default is 2
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate
    your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    anns = json.load(io.open(resFile, 'rb'))
    assert type(anns) == list, 'results is not an array of objects'
    question_ids = [ann['question_id'] for ann in anns]
    vqaEval.evaluate(set(question_ids))
    # print accuracies
    print "\n"
    print "Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall'])
    print "Per Question Type Accuracy is the following:"
    for quesType in vqaEval.accuracy['perQuestionType']:
        print "%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType])
    print "\n"
    print "Per Answer Type Accuracy is the following:"
    for ansType in vqaEval.accuracy['perAnswerType']:
        print "%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType])
    print "\n"
    json.dump(vqaEval.accuracy, io.open(score_json, 'wb'))
    # save evaluation results to ./Results folder
    # json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
    # json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
    # json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
    # json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))


def evaluate_dialog(predict_json, gt_json, score_json):
    """
    Not implemented
    """
    print "evaluate_dialog not implemented yet"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='training task', default='')
    parser.add_argument('--predict_json', help='prediction json', default='')
    parser.add_argument('--gt_json', help='groundtruth json', default='')
    parser.add_argument('--score_json', help='output score json', default='')
    args = parser.parse_args()

    if args.task == 'cap':
        evaluate_cap(args.predict_json, args.gt_json, args.score_json)
    elif args.task == 'qa':
        evaluate_qa(args.predict_json, args.gt_json, args.score_json)
    elif args.task == 'dialog':
        evaluate_dialog(args.predict_json, args.gt_json, args.score_json)
