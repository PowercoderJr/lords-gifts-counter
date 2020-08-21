import os
import time
import json
import argparse
import cv2
import pandas as pd
import pytesseract as pt


data_folder = 'data'
templates_folder = 'templates'
rarity_folder = 'rarity'
offsets_filename = 'offsets.json'
tmlp_gift_filename = 'gift.jpg'
tesseract_config = '-c tessedit_char_whitelist=" 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" --oem 3 --psm 7'
output_sheet_name = 'Main'
tm_threshold = 0.4
sender_name_margin = 5


def get_template_pos(frame, roi_ltrb, tmpl):
    roi = frame[roi_ltrb[1]:roi_ltrb[3], roi_ltrb[0]:roi_ltrb[2]]
    res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_pos, max_pos = cv2.minMaxLoc(res)
    ltrb = (
        roi_ltrb[0] + max_pos[0],
        roi_ltrb[1] + max_pos[1],
        roi_ltrb[0] + max_pos[0] + tmpl.shape[1],
        roi_ltrb[1] + max_pos[1] + tmpl.shape[0],
    )
    return max_val, ltrb, roi


def read_sender_name(frame, anchor_ltrb, right_border, margin):
    sender_fragment = frame[anchor_ltrb[1]-margin:anchor_ltrb[3]+margin, anchor_ltrb[2]:right_border]
    sender_fragment = cv2.threshold(sender_fragment, 127, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sender_fragment = cv2.morphologyEx(sender_fragment, cv2.MORPH_CLOSE, kernel)
    sender_fragment = cv2.blur(sender_fragment, (3, 3))
    sender = pt.image_to_string(sender_fragment, lang='eng', config=tesseract_config)[:-2]
    return sender, sender_fragment


def load_rarity_templates(path, scale):
    filenames = sorted(os.listdir(path), key=str.lower)
    tmpl_rarity = {}
    rarities = []
    for filename in filenames:
        tmpl = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        tmpl = cv2.threshold(tmpl, 127, 255, cv2.THRESH_BINARY_INV)[1]
        rarity_name = filename[filename.find(' ')+1:filename.rfind('.')]
        rarities.append(rarity_name)
        if scale != 1:
            tmpl = cv2.resize(tmpl, tuple([int(x * scale)
                for x in tmpl.shape[1::-1]]), cv2.INTER_AREA)
        tmpl_rarity[rarity_name] = tmpl
    return rarities, tmpl_rarity

def get_rarity(frame, roi_ltrb, templates):
    rarity_max_val = 0
    rarity = ''
    roi = frame[roi_ltrb[1]:roi_ltrb[3], roi_ltrb[0]:roi_ltrb[2]]
    roi_bin = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)[1]
    for r in templates:
        value, ltrb, _ = get_template_pos(roi_bin, [0, 0, roi.shape[1], roi.shape[0]], templates[r])
        #print(f'    {r}: {value:.2f}')
        if value > rarity_max_val:
            rarity_max_val = value
            rarity = r
    return rarity, ltrb, roi


def main(args):
    cap = cv2.VideoCapture(args.video_filename)
    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    resolution = f'{frame_w:.0f}x{frame_h:.0f}'
    print(f'Read video {resolution}')
    if not os.path.isdir(os.path.join(data_folder, resolution)):
        print('Error: unsupported resolution')
        print('Currently supported resolutions:')
        for item in os.listdir(data_folder):
            print(f'  {item}')
        cap.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    with open(os.path.join(data_folder, resolution, offsets_filename)) as json_file:
        offsets = json.load(json_file)
    frame_dur = int(1000 / fps)
    #print(f'{int(frame_w)}x{int(frame_h)}, FPS: {fps:.2f}, DUR: {frame_dur:.2f}')
    scaled_w = int(frame_w * args.scale)
    scaled_h = int(frame_h * args.scale)
    first_rois_ltrb = [[int(args.scale * offsets['first_rois_ltrb'][i][j]) for j in range(4)] for i in range(len(offsets['first_rois_ltrb']))]
    roi_ltrb = [int(args.scale * offsets['roi_ltrb'][j]) for j in range(4)]
    right_border = int(args.scale * offsets['right_border'])
    rarity_offset = [int(args.scale * offsets['rarity_offset'][j]) for j in range(4)]

    tmpl_gift = cv2.imread(os.path.join(data_folder, resolution, templates_folder, tmlp_gift_filename), cv2.IMREAD_GRAYSCALE)
    if scaled_w != frame_w:
        tmpl_gift = cv2.resize(tmpl_gift, tuple([int(x * args.scale)
            for x in tmpl_gift.shape[1::-1]]), cv2.INTER_AREA)
    rarities, tmpl_rarity = load_rarity_templates(os.path.join(data_folder, resolution, templates_folder, rarity_folder), args.scale)
    print('Data loaded successfully')
    print()

    senders = {}
    res, frame = cap.read()
    if scaled_w != frame_w:
        frame = cv2.resize(frame, (scaled_w, scaled_h))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if res:
        for f_roi_ltrb in first_rois_ltrb:
            value, t_ltrb, roi = get_template_pos(frame_gray, f_roi_ltrb, tmpl_gift)
            if value > tm_threshold:
                sender, sender_fragment = read_sender_name(frame_gray, t_ltrb,
                    right_border, sender_name_margin)
                rarity_roi_ltrb = (
                    t_ltrb[0] + rarity_offset[0],
                    t_ltrb[1] + rarity_offset[1],
                    t_ltrb[0] + rarity_offset[2],
                    t_ltrb[1] + rarity_offset[3],
                )
                rarity, _, _ = get_rarity(frame_gray, rarity_roi_ltrb, tmpl_rarity)
                if sender not in senders:
                    senders[sender] = {}
                senders[sender][rarity] = 1 if rarity not in senders[sender] else senders[sender][rarity] + 1
                print(f'{sender} ({rarity})')

    prev_top = 0
    frame_cap_time = 0
    while True:
        res, frame = cap.read()
        frame_cap_time = time.time()
        if not res: break
        if scaled_w != frame_w:
            frame = cv2.resize(frame, (scaled_w, scaled_h))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        value, t_ltrb, gift_roi = get_template_pos(frame_gray, roi_ltrb, tmpl_gift)
        if value > tm_threshold:
            if t_ltrb[1] > prev_top:
                sender, sender_fragment = read_sender_name(frame_gray, t_ltrb,
                    right_border, sender_name_margin)
                rarity_roi_ltrb = (
                    t_ltrb[0] + rarity_offset[0],
                    t_ltrb[1] + rarity_offset[1],
                    t_ltrb[0] + rarity_offset[2],
                    t_ltrb[1] + rarity_offset[3],
                )
                rarity, _, rarity_roi = get_rarity(frame_gray, rarity_roi_ltrb, tmpl_rarity)
                if sender not in senders:
                    senders[sender] = {}
                senders[sender][rarity] = 1 if rarity not in senders[sender] else senders[sender][rarity] + 1
                print(f'{sender} ({rarity})')
            prev_top = t_ltrb[1]

        if args.demonstration_mode > 0:
            cv2.rectangle(frame, (t_ltrb[0], t_ltrb[1]), (t_ltrb[2], t_ltrb[3]), (0, 200, 0) if value > tm_threshold else (0, 0, 200), 2)
            new_w = 1000
            cv2.imshow('video', cv2.resize(frame, (new_w, int(new_w / scaled_w * scaled_h))))
            key = cv2.waitKey(frame_dur)
            cv2.imshow('sender_fragment', sender_fragment)
            cv2.imshow('gift_roi', gift_roi)
            cv2.imshow('rarity_roi', rarity_roi)
            proc_dur = int((time.time() - frame_cap_time) * 1000)
            key = cv2.waitKey(frame_dur - proc_dur if frame_dur > proc_dur else 1) if args.demonstration_mode == 1 else cv2.waitKey()
            if key & 0xFF == 27: break
    cv2.destroyAllWindows()

    indices = []
    values = []
    for k, v in senders.items():
        indices.append(k)
        values.append([v.get(r, 0) for r in rarities])
    df = pd.DataFrame(values, index=indices, columns=rarities)
    df['Total'] = df.sum(axis='columns', numeric_only=True)
    df = df.sort_index(key=lambda x: x.str.lower())
    df = df.append(df.sum().rename('-- Total --'))
    writer = pd.ExcelWriter(args.output_filename)
    df.to_excel(writer, index_label='Nickname', sheet_name=output_sheet_name)
    writer.sheets[output_sheet_name].set_column('A:G', 15)
    workbook = writer.book
    worksheet = writer.sheets[output_sheet_name]
    border_format = workbook.add_format({'bottom':1, 'top':1, 'left':1, 'right':1})
    worksheet.conditional_format(f'A{len(indices) + 2}:G{len(indices) + 2}', {'type': 'no_errors', 'format': border_format})
    writer.save()
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Guild gifts counter for mobile game Lords Mobile for automatic accounting.')
    parser.add_argument('video_filename', type=str, help='input video')
    parser.add_argument('-s', '--scale', type=float, default=1.0, help='scale factor')
    parser.add_argument('-of', '--output_filename', default='senders.xlsx', type=str, help='output filename in *.xlsx format')
    parser.add_argument('-dm', '--demonstration_mode', choices=range(3), default=0, type=int,
        help='(0) - do not display video and roi, '\
             '(1) - display video and roi at normal speed (Esc to stop), '\
             '(2) - display video and roi frame by frame (any key to move on, Esc to stop)')

    args = parser.parse_args()
    main(args)
