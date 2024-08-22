import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt


def pixelate(image: np.array, num_pixels: np.array):
    pixel_dimensions = np.floor(image.shape[0:2] / num_pixels).astype(int)
    captured_area = num_pixels * pixel_dimensions
    image_reduced = np.zeros([num_pixels[0], num_pixels[1], 3])
    for v in range(0, pixel_dimensions[0]):
        for h in range(0, pixel_dimensions[1]):
            image_reduced += image[v:captured_area[0]:pixel_dimensions[0],
                                   h:captured_area[1]:pixel_dimensions[1], :]
    image_reduced /= np.prod(pixel_dimensions)
    image_reduced = image_reduced.astype(np.uint8)
    # Recreate full sized version of reduced image
    image_pixelated = np.zeros([captured_area[0], captured_area[1], 3], dtype=np.uint8)
    for v in range(0, pixel_dimensions[0]):
        for h in range(0, pixel_dimensions[1]):
            image_pixelated[v:captured_area[0]:pixel_dimensions[0],
                            h:captured_area[1]:pixel_dimensions[1], :] += image_reduced
    return image_reduced, image_pixelated, pixel_dimensions


def read_colour_data(colour_file_name):
    formatted_data = {}
    with open(colour_file_name, 'r') as file:
        data = file.readlines()
        headings = data[0].split('\n')[0].split('\t')
        for head_num in range(0, len(headings)):
            formatted_data[headings[head_num]] = []
            for line_num in range(1, len(data)):
                data_line_parts = data[line_num].split('\t')
                if len(data_line_parts) > 1:
                    entry = data_line_parts[head_num]
                    if 3 <= head_num <= 5:
                        entry = int(entry)
                    formatted_data[headings[head_num]].append(entry)
    colour_data_frame = pd.DataFrame(formatted_data)
    return colour_data_frame


def plot_colours(image: np.array, window_num=1):
    fig = plt.figure(window_num)
    ax = plt.axes(projection='3d')  # , elev=48, azim=134)
    # ax.set_position([0, 0, 0.95, 1])
    ax.scatter3D(image[:, :, 0],
                 image[:, :, 1],
                 image[:, :, 2],
                 c=image.reshape([-1, 3]) / 255)
    ax.set_xlabel('Blue')
    ax.set_ylabel('Green')
    ax.set_zlabel('Red')
    plt.show()
    return window_num + 1


def plot_pixel_counts(index_counts, colours, window_num):
    fig = plt.figure(window_num)
    ax = plt.axes()
    for c in range(0, colours.shape[0]):
        if c in index_counts.keys():
            ax.bar(c, index_counts[c], color=[colours[c][2]/255, colours[c][1]/255, colours[c][0]/255, 1])#, edgecolor=[0, 0, 0, 1])
    # ax.bar(np.arange(0, len(index_counts)), index_counts.values())
    plt.show()
    return window_num + 1


def enlarge_image(image: np.array, scale: np.array):
    enlarged_image = np.zeros(image.shape * np.array([scale[0], scale[1], 1]))
    for v in range(0, scale[0]):
        for h in range(0, scale[1]):
            enlarged_image[v::scale[0], h::scale[1], :] = image
    return enlarged_image


def get_symbol_list():
    symbol_group1 = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    symbol_group2 = '0123456789'
    symbol_list = []
    for i in symbol_group1:
        for j in symbol_group2:
            symbol_list.append(i + j)
    return symbol_list


def get_font_settings():
    # font = cv2.FONT_HERSHEY_SIMPLEX # alternative
    return {'font': cv2.FONT_HERSHEY_PLAIN,
            'scale': 0.7,
            'thickness': 1,
            'lineType': 2}


def generate_instruction(reduced_segment, index_array, colours_bgr, cell_width, bw):
    symbol_list = get_symbol_list()
    font_settings = get_font_settings()
    cell_scale = (cell_width * np.ones(2)).astype(int)
    labelled_segment = enlarge_image(reduced_segment, cell_scale)
    # black_white = np.array([[0, 0, 0], [255, 255, 255]])
    for v in range(0, index_array.shape[0]):
        for h in range(0, index_array.shape[1]):
            index = int(index_array[v, h])
            # cell_colour = colours_bgr[index, :]
            # bw = int(np.argmax(np.sum((black_white - cell_colour) ** 2, 1)))
            bottom_left_corner = np.array([h, v]) * cell_width + np.array([1, cell_width - 1 - 3])
            cv2.putText(labelled_segment, symbol_list[index], bottom_left_corner,
                        font_settings['font'], font_settings['scale'], 3 * [[0, 255][bw[index]]],
                        font_settings['thickness'], font_settings['lineType'])
    return labelled_segment


def format_sheet(sheet, v, h, border, cell_width, cell_margin):
    formatted_sheet = 255 * np.ones(sheet.shape + (2 * border + cell_margin) * np.array([1, 1, 0]))
    formatted_sheet[border:border + sheet.shape[0],
                    border:border + sheet.shape[1], :] = sheet
    for m in range(0, cell_margin):
        formatted_sheet[border + m:border + sheet.shape[0] + 1 + m:cell_width,
                        border:border + sheet.shape[1] + 1, :] = 0
        formatted_sheet[border:border + sheet.shape[0] + 1,
                        border + m:border + sheet.shape[1] + 1 + m:cell_width, :] = 0
    font_settings = get_font_settings()
    sheet_stitches = (np.array(sheet.shape[0:2]) / cell_width).astype(int)
    marker_period = 10
    marker_skip_stitches = (-1 * sheet_stitches * np.array([v, h])) % marker_period
    first_marker_offsets = border + marker_skip_stitches * cell_width
    num_markers = np.ceil((sheet_stitches - marker_skip_stitches) / marker_period).astype(int)
    for vh_index in [0, 1]:
        for marker_block in range(0, num_markers[vh_index]):
            marker_stitch = marker_block * marker_period
            marker_i_range = (first_marker_offsets[vh_index] +
                              cell_width * np.array([marker_stitch, marker_stitch + 1]))
            marker_1_range = border * np.ones(2) - np.array([cell_width, 0])
            marker_2_range = ((border + sheet.shape[1 - vh_index] + cell_margin) * np.ones(2) +
                              np.array([0, cell_width]))
            preceding_sheets = None
            marker_bounds = None
            marker_text_location = None
            if vh_index == 0:
                preceding_sheets = v
                marker_bounds = [[marker_i_range, marker_1_range],
                                 [marker_i_range, marker_2_range]]
                marker_text_location = [int(marker_i_range[1]), int(marker_2_range[1] + cell_width)]
            elif vh_index == 1:
                preceding_sheets = h
                marker_bounds = [[marker_1_range, marker_i_range],
                                 [marker_2_range, marker_i_range]]
                marker_text_location = [int(marker_1_range[0] - cell_width), int(marker_i_range[0])]
            for bounds in marker_bounds:
                formatted_sheet[int(bounds[0][0]):int(bounds[0][1]),
                                int(bounds[1][0]):int(bounds[1][1]), :] = 0
            marker_text_number = (preceding_sheets * sheet_stitches[vh_index] +
                                  marker_skip_stitches[vh_index] + marker_stitch)
            cv2.putText(formatted_sheet, f"{marker_text_number + 1}", marker_text_location[::-1],
                        font_settings['font'], font_settings['scale'] * 2, [0, 0, 0],
                        font_settings['thickness'], font_settings['lineType'])
    return formatted_sheet


def create_instruction_sheets(reduced_image, index_array, colours_bgr, parameters, bw):
    max_segment_size = parameters['max sheet size'][0:2] - 2 * parameters['border']
    cell_width = parameters['cell dim'] + parameters['cell margin']
    total_instruction_size = np.array(reduced_image.shape[0:2]) * cell_width
    num_sheets = np.ceil(total_instruction_size / max_segment_size).astype(int)
    print(f"number of sheets {num_sheets}")
    reduced_segment_size = np.ceil(reduced_image.shape[0:2] / num_sheets)
    formatted_sheets = {}
    for v in range(0, num_sheets[0]):
        v_span = (np.array([v, v + 1]) * reduced_segment_size[0]).astype(int)
        for h in range(0, num_sheets[1]):
            h_span = (np.array([h, h + 1]) * reduced_segment_size[1]).astype(int)
            reduced_segment = reduced_image[v_span[0]:v_span[1], h_span[0]:h_span[1], :]
            index_array_segment = index_array[v_span[0]:v_span[1], h_span[0]:h_span[1]]
            instruction_sheet = generate_instruction(reduced_segment, index_array_segment,
                                                     colours_bgr, cell_width, bw)
            # sheets[f"sheet_{v}_{h}"] = instruction_sheet
            formatted_sheets[f"sheet_{v}_{h}"] = format_sheet(instruction_sheet, v, h,
                                                              parameters['border'], cell_width,
                                                              parameters['cell margin'])
    return formatted_sheets


def make_conversion_table(index_counts, colours_names, colours_bgr, parameters, bw):
    max_table_dims = parameters['max sheet size'] + 2 * parameters['border'] * np.array([1, 1, 0])
    sample_cell_size = int(2 * (parameters['cell dim'] + parameters['cell margin']))
    sample_cell_gap = int(np.floor(sample_cell_size / 2))
    row_height = sample_cell_size + sample_cell_gap
    max_rows_per_column = int(np.floor(max_table_dims[0] / row_height))
    total_v_space = row_height * max_rows_per_column
    num_columns = 3
    num_tables = int(np.ceil(len(index_counts) / (max_rows_per_column * num_columns)))
    column_spacing = int(np.floor(max_table_dims[1] / num_columns))
    symbol_list = get_symbol_list()
    font_settings = get_font_settings()
    colour_indices = sorted(list(index_counts.keys()))
    entries = 0
    table_images = {}
    for t in range(0, num_tables):
        current_table_image = 255 * np.ones(max_table_dims)
        for c in range(0, num_columns):
            num_rows = min([len(index_counts) - entries, max_rows_per_column])
            h_offset = c * column_spacing
            # current_table_image[:, h_offset, :] = 0
            for r in range(0, num_rows):
                colour_index = colour_indices[entries]
                sample_colour = colours_bgr[[colour_index], :].reshape([1, 1, 3])
                cell_bounds_v = [r * row_height, r * row_height + sample_cell_size]
                cell_bounds_h = [h_offset, h_offset + sample_cell_size]
                # Fill colour
                current_table_image[cell_bounds_v[0]:cell_bounds_v[1],
                                    cell_bounds_h[0]:cell_bounds_h[1], :] = sample_colour
                # Draw enclosing box (1px)
                current_table_image[cell_bounds_v[0]:cell_bounds_v[1],
                                    cell_bounds_h, :] = 0
                current_table_image[cell_bounds_v,
                                    cell_bounds_h[0]:cell_bounds_h[1], :] = 0
                code_location = [cell_bounds_v[0] + int(np.floor(sample_cell_size * 3 / 4)),
                                 cell_bounds_h[0] + int(np.floor(sample_cell_size * 1 / 4))]
                DMC_location = [cell_bounds_v[0] + int(np.floor(sample_cell_size * 1 / 3)),
                                cell_bounds_h[1] + sample_cell_gap]
                name_location = [cell_bounds_v[0] + int(np.floor(sample_cell_size * 2 / 3)),
                                 cell_bounds_h[1] + sample_cell_gap]
                counts_location = [cell_bounds_v[1], cell_bounds_h[1] + sample_cell_gap]
                cv2.putText(current_table_image, symbol_list[colour_index], code_location[::-1],
                            font_settings['font'], font_settings['scale'], 3 * [[0, 255][bw[colour_index]]],
                            font_settings['thickness'], font_settings['lineType'])
                cv2.putText(current_table_image, f"DMC: {colours_names[colour_index, 0]}", DMC_location[::-1],
                            font_settings['font'], font_settings['scale'], [0, 0, 0],
                            font_settings['thickness'], font_settings['lineType'])
                cv2.putText(current_table_image, f"Name: {colours_names[colour_index, 1]}", name_location[::-1],
                            font_settings['font'], font_settings['scale'], [0, 0, 0],
                            font_settings['thickness'], font_settings['lineType'])
                cv2.putText(current_table_image, f"Stitches: {index_counts[colour_index]}", counts_location[::-1],
                            font_settings['font'], font_settings['scale'], [0, 0, 0],
                            font_settings['thickness'], font_settings['lineType'])
                entries += 1
        table_images[t] = current_table_image
    return table_images


def main(image_file_name, dimensions_inches, stitches_per_inch, colour_data, parameters):
    # Retrieve and display raw image.
    picture = cv2.imread(image_file_name)
    cv2.imshow("Original image", picture)
    cv2.waitKey(0)
    # Calculate number of stitches that will be used.
    num_stitches = np.round(stitches_per_inch * dimensions_inches).astype(int)
    print(f"Number of stitches: {num_stitches[0]} (v) X {num_stitches[1]} (h)")
    # Create and display pixelated image.
    picture_reduced, picture_pixelated, pixel_dims = pixelate(picture, num_stitches)
    cv2.imshow("Pixelated image", picture_pixelated)
    cv2.waitKey(0)
    # Match colours generated via pixelation to most similar thread colour.
    colours_bgr = np.array(colour_data[['Blue', 'Green', 'Red']])
    matched_picture_reduced = np.zeros(picture_reduced.shape)
    matched_index_array = np.zeros(picture_reduced.shape[0:2])
    matched_index_counts = {}
    for v in range(0, picture_reduced.shape[0]):
        for h in range(0, picture_reduced.shape[1]):
            pixel_bgr = picture_reduced[[v], [h], :].reshape([-1, 3])
            differences_bgr = colours_bgr - np.tile(pixel_bgr, [colours_bgr.shape[0], 1])
            distances_bgr = np.sum(differences_bgr ** 2, 1)
            closest_index = np.argmin(distances_bgr)
            matched_picture_reduced[[v], [h], :] = colours_bgr[[closest_index], :].reshape([1, 1, 3])
            matched_index_array[v, h] = closest_index
            if closest_index in matched_index_counts.keys():
                matched_index_counts[closest_index] += 1
            else:
                matched_index_counts[closest_index] = 1
    print(matched_index_counts)
    print(len(matched_index_counts))
    # Display pixelated image with colours replaced with matched thread colours.
    cv2.imshow("matched picture", matched_picture_reduced.astype(np.uint8))

    # Assign black / white text to overlay each thread colour
    black_white = np.array([[0, 0, 0], [255, 255, 255]])
    bw = [int(np.argmax(np.sum((black_white - colours_bgr[index, :]) ** 2, 1)))
          for index in range(0, colours_bgr.shape[0])]

    # Generate instruction patterns
    instructions = create_instruction_sheets(matched_picture_reduced, matched_index_array,
                                             colours_bgr, parameters, bw)
    for sheet_name in instructions.keys():
        cv2.imshow(f"Formatted instructions {sheet_name}", instructions[sheet_name].astype(np.uint8))
        cv2.imwrite(f"instructions {sheet_name}.png", instructions[sheet_name].astype(np.uint8))
        cv2.waitKey(0)

    # Display statistical distributions
    fig_num = 1
    fig_num = plot_colours(matched_picture_reduced, fig_num)
    fig_num = plot_pixel_counts(matched_index_counts, colours_bgr, fig_num)

    # Create legend tables
    colours_names = np.array(colour_data[['DMC', 'Floss Name']])
    conversion_tables = make_conversion_table(matched_index_counts, colours_names,
                                              colours_bgr, parameters, bw)
    for t, table in conversion_tables.items():
        cv2.imshow(f"Conversion table {t}", table.astype(np.uint8))
        cv2.waitKey(0)
        cv2.imwrite(f"instructions table_{t}.png", table.astype(np.uint8))
    return 0


if __name__ == "__main__":
    picture_file_name = "olliePlusAbbey.jpg"
    # picture_file_name = "example_picture_lowres.png"
    inches = np.array([14, 12])
    sizes = [16, 14, 12]
    num_stitches_per_inch = sizes[1]

    all_colour_data = read_colour_data('colourData.txt')git

    a4_dims = np.array([3508, 2480, 3])
    main(picture_file_name, inches, num_stitches_per_inch, all_colour_data,
         parameters={'cell dim': 15,
                     'cell margin': 1,
                     'max sheet size': np.array([2000, 1200, 3]),
                     'border': 100})
