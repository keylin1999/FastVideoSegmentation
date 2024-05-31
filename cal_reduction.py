import argparse

total_mul_add = 6062723825664

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--log_file', type=str)
    args.add_argument('-ps', action='store_true', default=False)
    args = args.parse_args()
    if args.ps:
        total_mul_add = 87842504048640

    text = ""
    with open('output/resnet18_d3/' + args.log_file + '/evaluation.txt', 'r') as f:
        text = f.read()

    text = text.split('\n')
    for line in text:
        if 'Total ops:' in line:
            text = line
            break
    
    text = text[12:-1]
    pre, post = text.split(', ')
    pre = float(pre)
    post = float(post)

    # rewrite following to print precision up to 4 decimal places
    print(f"pre: {pre/total_mul_add:.4f} post: {post/total_mul_add:.4f} reduction: {1 - (pre + post) / total_mul_add:.4f} percentage: {(pre + post) / total_mul_add:.4f}")