import os
import json
import numpy as np
from Simulation.ik_sim.script_sim import calc_ik, plot_results

RECORDINGS = [
    ('bellboy-robotics/B-br_open_door-20251130-201101-BILLIE-11',           'hilton_berlin/br_open_door'),
    ('bellboy-robotics/B-bl_open_door-20251204-161829-BILLIE-11',           'hilton_berlin/bl_open_door'),
    ('bellboy-robotics/B-unknown-20260507-144051-BILLIE-14',                'seminaris_berlin/door_bl'),
    ('bellboy-robotics/B-unknown-20260507-170550-BILLIE-14',                'seminaris_berlin/door_br'),  
    ('bellboy-robotics/B-unknown-20260225-185842-BILLIE-12',                'marriott_rome/entrance_bl'),
    ('bellboy-robotics/B-unknown-20260301-180408-BILLIE-12',                'marriott_rome/entrance_br'),


    # ('bellboy-robotics/B-Cabinet1-BR-20260218-084952-BILLIE-15',            'hilton_berlin/cabinet1_br'),
    # ('bellboy-robotics/B-bathroom_shower_policy-20260107-202727-BILLIE-14', 'seminaris_berlin/bathroom_shower_policy_bl'),
    # ('bellboy-robotics/B-below_the_bed-20260105-180706-BILLIE-14',          'seminaris_berlin/below_the_bed_br'),
    # ('bellboy-robotics/B-bl_close_bathroom-20251116-160617-BILLIE-11',      'hilton_berlin/bl_close_bathroom'),
    # ('bellboy-robotics/B-bl_close_cabinet_1-20251116-142328-BILLIE-11',     'hilton_berlin/bl_close_cabinet_1'),
    # ('bellboy-robotics/B-bl_close_tv_drawer-20251103-164231-BILLIE-11',     'hilton_berlin/bl_close_tv_drawer'),
    # ('bellboy-robotics/B-bl_fold_from_under_bed-20251109-145145-BILLIE-11', 'hilton_berlin/bl_fold_from_under_bed'),
    # ('bellboy-robotics/B-bl_shut_the_door-20251116-161448-BILLIE-11',       'hilton_berlin/bl_shut_the_door'),
    # ('bellboy-robotics/B-bl_under_bed-20251109-144241-BILLIE-11',           'hilton_berlin/bl_under_bed'),
    # ('bellboy-robotics/B-br_close_bathroom_door-20251117-091725-BILLIE-11', 'hilton_berlin/br_close_bathroom_door'),
    # ('bellboy-robotics/B-br_fold_from_under_bed-20251105-184501-BILLIE-11', 'hilton_berlin/br_fold_from_under_bed'),
    # ('bellboy-robotics/B-br_shut_the_door-20251117-092037-BILLIE-11', 'hilton_berlin/br_shut_the_door'),
    # ('bellboy-robotics/B-br_superior_bathtub_out-20260106-152201-BILLIE-14', 'seminaris_berlin/br_superior_bathtub_out'),
    # ('bellboy-robotics/B-br_superor_bathub-20260106-151817-BILLIE-14', 'seminaris_berlin/br_superor_bathub'),
    # ('bellboy-robotics/B-br_under_bed-20251105-184210-BILLIE-11', 'hilton_berlin/br_under_bed'),
    # ('bellboy-robotics/B-cabinet-1-BL-20260217-164034-BILLIE-11', 'hilton_berlin/cabinet_1_bl'),
    # ('bellboy-robotics/B-close_cabinet-20260105-182935-BILLIE-14', 'seminaris_berlin/close_cabinet_br_1'),
    # ('bellboy-robotics/B-close_cabinet-20260105-183024-BILLIE-14', 'seminaris_berlin/close_cabinet_br_2'),
    # ('bellboy-robotics/B-close_tv_drawer_bt-20251102-154439-BILLIE-11', 'hilton_berlin/close_tv_drawer_bt_br'),
    # ('bellboy-robotics/B-coffee-drawer-br-20260115-084428-BILLIE-12', 'marriott_rome/coffee_drawer_br'),
    # ('bellboy-robotics/B-fold_from_under_bed-20260107-172419-BILLIE-14', 'seminaris_berlin/fold_from_under_bed_bl'),
    # ('bellboy-robotics/B-open-office-cabinet-20251228-114500-BILLIE-17', 'office_telaviv/open_office_cabinet_2'),
    # ('bellboy-robotics/B-open-toilet-20251216-101344-BILLIE-17', 'office_telaviv/open_toilet'),
    # ('bellboy-robotics/B-open_office_cabinet-20251210-094956-BILLIE-17', 'office_telaviv/open_office_cabinet_1'),
    # ('bellboy-robotics/B-policy_record-20260125-193134-BILLIE-12', 'marriott_rome/exit_room_bl_1'),
    # ('bellboy-robotics/B-rat-chair-for-bump-20260205-091910-BILLIE-17', 'office_telaviv/rat_chair_for_bump'),
    # ('bellboy-robotics/B-standard_bath_br_close_toilet-20260202-092703-BILLIE-14', 'seminaris_berlin/standard_bath_br_close_toilet'),
    # ('bellboy-robotics/B-standard_bath_br_open_toilet-20260202-092547-BILLIE-14', 'seminaris_berlin/standard_bath_br_open_toilet'),
    # ('bellboy-robotics/B-standard_bl_bath_close_toilet-20260121-143859-BILLIE-14', 'seminaris_berlin/standard_bl_bath_close_toilet'),
    # ('bellboy-robotics/B-standard_bl_bath_open_toilet-20260121-143524-BILLIE-14', 'seminaris_berlin/standard_bl_bath_open_toilet'),
    # ('bellboy-robotics/B-standard_bl_close_cabinet_1-20260217-115654-BILLIE-14', 'seminaris_berlin/standard_bl_close_cabinet_1'),
    # ('bellboy-robotics/B-standard_bl_close_cabinet_2-20260217-115831-BILLIE-14', 'seminaris_berlin/standard_bl_close_cabinet_2'),
    # ('bellboy-robotics/B-standard_bl_close_door_1-20260121-153002-BILLIE-14', 'seminaris_berlin/standard_bl_close_door_1'),
    # ('bellboy-robotics/B-standard_bl_close_door_2-20260121-153359-BILLIE-14', 'seminaris_berlin/standard_bl_close_door_2_1'),
    # ('bellboy-robotics/B-standard_bl_close_door_2-20260202-131901-BILLIE-14', 'seminaris_berlin/standard_bl_close_door_2_2'),
    # ('bellboy-robotics/B-standard_bl_close_door_2-20260203-113215-BILLIE-14', 'seminaris_berlin/standard_bl_close_door_2_3'),
    # ('bellboy-robotics/B-standard_bl_close_door_3-20260121-160418-BILLIE-14', 'seminaris_berlin/standard_bl_close_door_3_1'),
    # ('bellboy-robotics/B-standard_bl_close_door_3-20260203-113530-BILLIE-14', 'seminaris_berlin/standard_bl_close_door_3_2'),
    # ('bellboy-robotics/B-standard_bl_closet_open_cabinet-20260209-113550-BILLIE-14', 'seminaris_berlin/standard_bl_closet_open_cabinet'),
    # ('bellboy-robotics/B-standard_bl_in_shower-20260112-164655-BILLIE-14', 'seminaris_berlin/standard_bl_in_shower'),
    # ('bellboy-robotics/B-standard_bl_into_bath-20260111-170027-BILLIE-14', 'seminaris_berlin/standard_bl_into_bath'),
    # ('bellboy-robotics/B-standard_bl_into_toilet-20260111-162436-BILLIE-14', 'seminaris_berlin/standard_bl_into_toilet'),
    # ('bellboy-robotics/B-standard_bl_out_of_bath-20260111-171158-BILLIE-14', 'seminaris_berlin/standard_bl_out_of_bath'),
    # ('bellboy-robotics/B-standard_bl_out_shower-20260112-165830-BILLIE-14', 'seminaris_berlin/standard_bl_out_shower'),
    # ('bellboy-robotics/B-standard_bl_shower_close_toilet-20260112-172758-BILLIE-14', 'seminaris_berlin/standard_bl_shower_close_toilet'),
    # ('bellboy-robotics/B-standard_bl_shower_open_toilet-20260112-172215-BILLIE-14', 'seminaris_berlin/standard_bl_shower_open_toilet'),
    # ('bellboy-robotics/B-standard_br_close_door_1-20260114-124539-BILLIE-14', 'seminaris_berlin/standard_br_close_door_1_1'),
    # ('bellboy-robotics/B-standard_br_close_door_1-20260128-132455-BILLIE-14', 'seminaris_berlin/standard_br_close_door_1_2'),
    # ('bellboy-robotics/B-standard_br_close_door_2-20260201-192539-BILLIE-14', 'seminaris_berlin/standard_br_close_door_2_1'),
    # ('bellboy-robotics/B-standard_br_close_door_2-20260202-093143-BILLIE-14', 'seminaris_berlin/standard_br_close_door_2_2'),
    # ('bellboy-robotics/B-standard_br_close_door_3-20260202-093431-BILLIE-14', 'seminaris_berlin/standard_br_close_door_3'),
    # ('bellboy-robotics/B-standard_br_close_toilet-20260201-123439-BILLIE-14', 'seminaris_berlin/standard_br_close_toilet'),
    # ('bellboy-robotics/B-standard_br_fold_from_under_bed-20260128-200205-BILLIE-14', 'seminaris_berlin/standard_br_fold_from_under_bed'),
    # ('bellboy-robotics/B-standard_br_into_shower-20260129-200702-BILLIE-14', 'seminaris_berlin/standard_br_into_shower'),
    # ('bellboy-robotics/B-standard_br_open_toilet-20260201-123318-BILLIE-14', 'seminaris_berlin/standard_br_open_toilet'),
    # ('bellboy-robotics/B-standard_br_out_of_shower-20260129-202019-BILLIE-14', 'seminaris_berlin/standard_br_out_of_shower'),
    # ('bellboy-robotics/B-standard_br_under_bed-20260128-200037-BILLIE-14', 'seminaris_berlin/standard_br_under_bed'),
    # ('bellboy-robotics/B-standard_closet_bl_close_cabinet-20260209-113849-BILLIE-14', 'seminaris_berlin/standard_closet_bl_close_cabinet'),
    # ('bellboy-robotics/B-standard_closet_br_close_cabinet-20260201-204647-BILLIE-14', 'seminaris_berlin/standard_closet_br_close_cabinet'),
    # ('bellboy-robotics/B-standard_closet_br_open_cabinet-20260201-204438-BILLIE-14', 'seminaris_berlin/standard_closet_br_open_cabinet'),
    # ('bellboy-robotics/B-standard_plus_bl_close_bathroom_door-20260213-113628-BILLIE-14', 'seminaris_berlin/standard_plus_bl_close_bathroom_door'),
    # ('bellboy-robotics/B-standard_plus_bl_close_toilet-20260213-113345-BILLIE-14', 'seminaris_berlin/standard_plus_bl_close_toilet'),
    # ('bellboy-robotics/B-standard_plus_bl_close_toilet_door_2-20260213-113826-BILLIE-14', 'seminaris_berlin/standard_plus_bl_close_toilet_door_2'),
    # ('bellboy-robotics/B-standard_plus_bl_open_toilet-20260213-113208-BILLIE-14', 'seminaris_berlin/standard_plus_bl_open_toilet'),
    # ('bellboy-robotics/B-standard_plus_br_close_toilet-20260201-172013-BILLIE-14', 'seminaris_berlin/standard_plus_br_close_toilet'),
    # ('bellboy-robotics/B-standard_plus_br_close_toilet_door_1-20260201-172249-BILLIE-14', 'seminaris_berlin/standard_plus_br_close_toilet_door_1'),
    # ('bellboy-robotics/B-standard_plus_br_close_toilet_door_2-20260201-172447-BILLIE-14', 'seminaris_berlin/standard_plus_br_close_toilet_door_2'),
    # ('bellboy-robotics/B-standard_plus_br_into_shower-20260201-171106-BILLIE-14', 'seminaris_berlin/standard_plus_br_into_shower'),
    # ('bellboy-robotics/B-standard_plus_br_open_toilet-20260201-171922-BILLIE-14', 'seminaris_berlin/standard_plus_br_open_toilet'),
    # ('bellboy-robotics/B-standard_plus_br_out_of_shower-20260201-171405-BILLIE-14', 'seminaris_berlin/standard_plus_br_out_of_shower'),
    # ('bellboy-robotics/B-superior_bl_bathtub_open_toilet-20260115-162557-BILLIE-14', 'seminaris_berlin/superior_bl_bathtub_open_toilet'),
    # ('bellboy-robotics/B-superior_bl_close_door_1-20260120-163412-BILLIE-14', 'seminaris_berlin/superior_bl_close_door_1'),
    # ('bellboy-robotics/B-superior_bl_close_door_2-20260120-170522-BILLIE-14', 'seminaris_berlin/superior_bl_close_door_2'),
    # ('bellboy-robotics/B-superior_bl_close_door_3-20260203-104700-BILLIE-14', 'seminaris_berlin/superior_bl_close_door_3'),
    # ('bellboy-robotics/B-superior_bl_enter_bathtub-20260115-181444-BILLIE-14', 'seminaris_berlin/superior_bl_enter_bathtub'),
    # ('bellboy-robotics/B-superior_bl_into_shower-20260120-160754-BILLIE-14', 'seminaris_berlin/superior_bl_into_shower'),
    # ('bellboy-robotics/B-superior_bl_out_of_bath-20260202-155823-BILLIE-14', 'seminaris_berlin/superior_bl_out_of_bath'),
    # ('bellboy-robotics/B-superior_bl_out_shower-20260120-161715-BILLIE-14', 'seminaris_berlin/superior_bl_out_shower'),
    # ('bellboy-robotics/B-superior_bl_shower_close_toilet-20260202-114330-BILLIE-14', 'seminaris_berlin/superior_bl_shower_close_toilet'),
    # ('bellboy-robotics/B-superior_bl_shower_open_toilet-20260202-114236-BILLIE-14', 'seminaris_berlin/superior_bl_shower_open_toilet'),
    # ('bellboy-robotics/B-superior_bl_shower_underbed-20260107-172318-BILLIE-14', 'seminaris_berlin/superior_bl_shower_underbed'),
    # ('bellboy-robotics/B-superior_br_bath_close_toilet-20260115-163207-BILLIE-14', 'seminaris_berlin/superior_br_bath_close_toilet'),
    # ('bellboy-robotics/B-superior_br_into_shower-20260106-122417-BILLIE-14', 'seminaris_berlin/superior_br_into_shower_1'),
    # ('bellboy-robotics/B-superior_br_into_shower-20260120-133642-BILLIE-14', 'seminaris_berlin/superior_br_into_shower_2'),
    # ('bellboy-robotics/B-superior_br_out_shower-20260120-134238-BILLIE-14', 'seminaris_berlin/superior_br_out_shower'),
    # ('bellboy-robotics/B-superoir_br_close_door_2-20260118-182452-BILLIE-14', 'seminaris_berlin/superoir_br_close_door_2'),
    # ('bellboy-robotics/B-under_bed_bl_std-20260108-154239-BILLIE-14', 'seminaris_berlin/under_bed_bl_std'),
    # ('bellboy-robotics/B-under_the_bed_up-20260105-180847-BILLIE-14', 'seminaris_berlin/under_the_bed_up_br'),
    # ('bellboy-robotics/B-unknown-20250810-073207-BILLIE-01', 'office_telaviv/slide'),
    # ('bellboy-robotics/B-unknown-20251104-130759-BILLIE-12', 'marriott_rome/bed_2_suite_bl'),
    # ('bellboy-robotics/B-unknown-20251104-134131-BILLIE-12', 'marriott_rome/bathroom_suite_bl'),
    # ('bellboy-robotics/B-unknown-20251104-134758-BILLIE-12', 'marriott_rome/bathroom_shower_overall_suite_bl'),
    # ('bellboy-robotics/B-unknown-20251105-131907-BILLIE-12', 'marriott_rome/bed_2_suite_br'),
    # ('bellboy-robotics/B-unknown-20251105-154024-BILLIE-12', 'marriott_rome/bathroom_suite_br'),
    # ('bellboy-robotics/B-unknown-20251105-154443-BILLIE-12', 'marriott_rome/bathroom_shower_overall_suite_br'),
    # ('bellboy-robotics/B-unknown-20251210-134842-BILLIE-17', 'office_telaviv/toilet'),
    # ('bellboy-robotics/B-unknown-20260107-202948-BILLIE-14', 'seminaris_berlin/bathroom_towel_rack_2_bl'),
    # ('bellboy-robotics/B-unknown-20260107-203025-BILLIE-14', 'seminaris_berlin/bathroom_toilet_area_bl'),
    # ('bellboy-robotics/B-unknown-20260111-121431-BILLIE-12', 'marriott_rome/coffee_drawer_bl'),
    # ('bellboy-robotics/B-unknown-20260119-114412-BILLIE-14', 'seminaris_berlin/exit_bathroom_br'),
    # ('bellboy-robotics/B-unknown-20260125-205448-BILLIE-08', 'marriott_rome/exit_room_br_1'),
    # ('bellboy-robotics/B-unknown-20260125-221409-BILLIE-08', 'marriott_rome/exit_room_br_2'),
    # ('bellboy-robotics/B-unknown-20260301-195655-BILLIE-15', 'hilton_berlin/desk_area_drawer_bl'),
    # ('bellboy-robotics/B-unknown-20260303-163305-BILLIE-11', 'hilton_berlin/cabinet_1_items_br'),
    # ('bellboy-robotics/B-unknown-20260303-171428-BILLIE-11', 'hilton_berlin/desk_area_drawer_br'),
    # ('bellboy-robotics/B-unknown-20260326-164816-BILLIE-11', 'hilton_berlin/bathroom_br'),
    # ('bellboy-robotics/B-unknown-20260407-181059-BILLIE-14', 'seminaris_berlin/cabinet_3_bl_1'),
    # ('bellboy-robotics/B-unknown-20260408-101838-BILLIE-14', 'seminaris_berlin/cabinet_3_br'),
    # ('bellboy-robotics/B-unknown-20260412-183445-BILLIE-12', 'marriott_rome/toilet_suite_br_1'),
    # ('bellboy-robotics/B-unknown-20260412-184112-BILLIE-12', 'marriott_rome/toilet_suite_br_2'),
    # ('bellboy-robotics/B-unknown-20260413-092243-BILLIE-14', 'seminaris_berlin/cabinet_3_bl_2'),
    # ('bellboy-robotics/B-unknown-20260414-154029-BILLIE-14', 'seminaris_berlin/bathroom_toilet_brush'),
    # ('bellboy-robotics/B-unknown-20260414-154307-BILLIE-14', 'seminaris_berlin/bathroom_toilet_area_2_br'),
    # ('bellboy-robotics/B-unknown-20260416-152552-BILLIE-11', 'hilton_berlin/bathroom_bl'),

    # ('bellboy-robotics/B-unknown-20260510-133203-BILLIE-12', 'marriott_rome/exit_room_br_3'),
    # ('bellboy-robotics/B-unknown-20260511-074202-BILLIE-12', 'marriott_rome/exit_room_bl_2'),
    # ('bellboy-robotics/B-unknown-20260511-074318-BILLIE-12', 'marriott_rome/exit_room_bl_3'),
    # ('bellboy-robotics/B-unknown-20260517-122533-BILLIE-08', 'marriott_rome/glass_door_suite'),
    # ('bellboy-robotics/B-unknown-20260517-132311-BILLIE-08', 'marriott_rome/bathroom_door_suite_br'),
    # ('bellboy-robotics/B-unknown-20260518-135104-BILLIE-08', 'marriott_rome/bathroom_door_suite_bl'),
    # ('bellboy-robotics/B-unknown-20260604-112546-BILLIE-08', 'marriott_rome/toilet_suite_bl_1'),
    # ('bellboy-robotics/B-unknown-20260604-114346-BILLIE-08', 'marriott_rome/toilet_suite_bl_2'),
    # ('bellboy-robotics/B-unknown-20260604-115309-BILLIE-08', 'marriott_rome/toilet_suite_bl_3'),
    # ('bellboy-robotics/B-unknown-20260618-114305-BILLIE-20', 'holiday_inn_berlin/bathroom_bl_1'),
    # ('bellboy-robotics/B-unknown-20260618-115559-BILLIE-20', 'holiday_inn_berlin/bathroom_bl_2'),
    # ('bellboy-robotics/B-unknown-20260618-120320-BILLIE-20', 'holiday_inn_berlin/bathroom_shower_head_bl'),
    # ('bellboy-robotics/B-unknown-20260618-120631-BILLIE-20', 'holiday_inn_berlin/bathroom_towel_rack_bl'),
    # ('bellboy-robotics/B-unknown-20260618-124501-BILLIE-20', 'holiday_inn_berlin/bathroom_toilet_seat_cover_bl'),
    # ('bellboy-robotics/B-unknown-20260618-125450-BILLIE-20', 'holiday_inn_berlin/bathroom_toilet_bowl_bl'),
]

# Total: 135 recordings

_OUTPUT_BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'recordings')

def run_recording(repo_id, name, arm_sim_type, ik_solver, use_curr_joints, n_random_starts):
    output_dir = os.path.join(_OUTPUT_BASE, f'{name}_starts{n_random_starts}')
    os.makedirs(output_dir, exist_ok=True)

    params = {
        'repo_id':         repo_id,
        'name':            name,
        'arm_sim_type':    arm_sim_type,
        'ik_solver':       ik_solver,
        'use_curr_joints': use_curr_joints,
        'n_random_starts': n_random_starts,
    }

    print(f'\n{"=" * 60}')
    print(f'Recording : {name}')
    print(f'repo_id   : {repo_id}')
    print(f'params    : sim={arm_sim_type}  ik={ik_solver}  joints={use_curr_joints}  starts={n_random_starts}')
    print(f'{"=" * 60}')

    try:
        expected_poses, new_pos, problematic_frames = calc_ik(
            arm_sim_type, ik_solver, use_curr_joints, n_random_starts, repo_id
        )
    except Exception as e:
        print(f'ERROR: {e}')
        with open(os.path.join(output_dir, 'error.txt'), 'w') as f:
            f.write(str(e))
        with open(os.path.join(output_dir, 'params.json'), 'w') as f:
            json.dump(params, f, indent=2)
        return

    np.save(os.path.join(output_dir, 'sim_expected_poses.npy'), expected_poses)
    np.save(os.path.join(output_dir, 'sim_new_pos.npy'), new_pos)
    with open(os.path.join(output_dir, 'sim_problematic_frames.json'), 'w') as f:
        json.dump([{str(k): str(v) for k, v in frame.items()} for frame in problematic_frames], f, indent=2)
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    diff = new_pos - expected_poses
    print(f'Mean error pos: X={np.mean(diff[:, 0]):.2f}  Y={np.mean(diff[:, 1]):.2f}  Z={np.mean(diff[:, 2]):.2f} mm')
    print(f'Mean error ori: rx={np.mean(diff[:, 3]):.2f}  ry={np.mean(diff[:, 4]):.2f}  rz={np.mean(diff[:, 5]):.2f} deg')

    title = f'{name} | {arm_sim_type} | ik={ik_solver} | joints={use_curr_joints} | starts={n_random_starts}'
    html_name = name.replace('/', '_') + f'_starts{n_random_starts}.html'
    plot_results(expected_poses, new_pos, problematic_frames,
                 title=title,
                 output_path=os.path.join(output_dir, html_name))

    print(f'Saved to {output_dir}')


def print_summary():
    rows = []
    for root, dirs, _ in os.walk(_OUTPUT_BASE):
        dirs.sort()
        exp_path = os.path.join(root, 'sim_expected_poses.npy')
        new_path = os.path.join(root, 'sim_new_pos.npy')
        err_path = os.path.join(root, 'error.txt')
        entry    = os.path.relpath(root, _OUTPUT_BASE)
        if entry == '.':
            continue
        if os.path.exists(err_path) and not os.path.exists(new_path):
            rows.append((entry, None, None, None, None, None, 'ERROR'))
            continue
        if not (os.path.exists(exp_path) and os.path.exists(new_path)):
            continue
        expected = np.load(exp_path)
        actual   = np.load(new_path)
        n = min(len(expected), len(actual))
        diff = actual[:n] - expected[:n]
        ori_diff = ((diff[:, 3:] + 180) % 360) - 180   # wrap to [-180, 180]
        pos_norm = np.linalg.norm(diff[:, :3], axis=1)
        ori_norm = np.linalg.norm(ori_diff,    axis=1)
        max_pos, max_ori = float(np.max(pos_norm)), float(np.max(ori_norm))
        if max_pos < 1.0 and max_ori < 3.0:
            continue
        rows.append((entry, n, float(np.mean(pos_norm)), max_pos,
                               float(np.mean(ori_norm)), max_ori, 'ok'))

    if not rows:
        print('No results found.')
        return

    def get_location(dir_name):
        params_path = os.path.join(_OUTPUT_BASE, dir_name, 'params.json')
        if os.path.exists(params_path):
            with open(params_path) as f:
                p = json.load(f)
            rec_name = p.get('name', dir_name)
            return rec_name.split('/')[0] if '/' in rec_name else 'other'
        return 'other'

    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        groups[get_location(row[0])].append(row)

    name_w = max(len(r[0]) for r in rows)
    header = f'{"Recording":<{name_w}}  {"Frames":>6}  {"MeanPos":>8}  {"MaxPos":>8}  {"MeanOri":>8}  {"MaxOri":>8}  Status'
    sep    = '-' * len(header)

    for location in sorted(groups):
        print(f'\n── {location} ' + '─' * max(0, len(sep) - len(location) - 4))
        print(header)
        print(sep)
        for row in groups[location]:
            name, status = row[0], row[-1]
            if status == 'ERROR':
                print(f'{name:<{name_w}}  {"—":>6}  {"—":>8}  {"—":>8}  {"—":>8}  {"—":>8}  ERROR')
            else:
                _, n, mean_pos, max_pos, mean_ori, max_ori, _ = row
                print(f'{name:<{name_w}}  {n:>6}  {mean_pos:>8.2f}  {max_pos:>8.2f}  {mean_ori:>8.2f}  {max_ori:>8.2f}  ok')


if __name__ == '__main__':
    arm_sim_type    = 'pybullet'
    ik_solver       = 'pyroki'
    use_curr_joints = 'actual'
    n_random_starts = 0

    'set seed'
    seed = np.random.randint(0, 2**31)
    print(f"[ik_pyroki] random seed: {seed}")
    np.random.seed(seed)

    for repo_id, name in RECORDINGS:
        rng_state = np.random.get_state()
        print(f"rand start {int(rng_state[2])}")
        n_random_starts = 0
        run_recording(repo_id, name, arm_sim_type, ik_solver, use_curr_joints, n_random_starts)
        # rng_state = np.random.get_state()
        # print(f"rand start {int(rng_state[2])}")
        # n_random_starts = 2
        # run_recording(repo_id, name, arm_sim_type, ik_solver, use_curr_joints, n_random_starts)

    print_summary()
    