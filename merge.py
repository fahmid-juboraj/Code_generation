import json

def merge_json_responses(submission_json_path, sub_406_json_path, output_json_path, target_ids):
    with open(submission_json_path, 'r', encoding='utf-8') as f:
        submission_data = json.load(f)
    with open(sub_406_json_path, 'r', encoding='utf-8') as f:
        sub_406_data = json.load(f)

    sub_406_dict = {item['id']: item['response'] for item in sub_406_data}

    for item in submission_data:
        if item['id'] in target_ids and item['id'] in sub_406_dict:
            item['response'] = sub_406_dict[item['id']]

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(submission_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    submission_json_path = 'submission.json'
    sub_406_json_path = '/content/corrected_from_error.json'
    output_json_path = 'merged_submission.json'
    target_ids = [ ] #put the target ids for merging with the passed test cases

    merge_json_responses(submission_json_path, sub_406_json_path, output_json_path, target_ids)
