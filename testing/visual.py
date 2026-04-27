import numpy as np

import pygame

import network.neuralNet as NN

def visual_test(net: NN.NeuralNet, dataset: str, find_incorrect: bool, rand_order: bool):

    
    class_names= [] #to avoid possibly unbound errors
    
    if dataset == "CIFAR10":
        # import visualize.visualizeCIFAR as VM
        import setup_datasets.CIFAR10 as DATA
        class_names = DATA.class_names

    # else:
    #     import visualize.visualizeMNIST as VM
    elif dataset == "MNIST":
        # if dataset == "MNIST":
    
        import setup_datasets.MNIST as DATA
    else:
        import setup_datasets.FASHION_MNIST as DATA
        class_names = DATA.class_names
    import visualize.draw as draw

    is_RGB = dataset == "CIFAR10"
    
    
    x_test = DATA.x_test
    y_test = DATA.y_test

    if rand_order:
        idx = np.random.permutation(x_test.shape[0])
        
        x_test = x_test[idx]
        y_test = y_test[idx]
    
    index = -1
    has_changed = True

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT}))
    testing = True
    while testing:
        draw.clock.tick(draw.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                testing = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and index < len(x_test)-1:
                    index += 1
                    has_changed = True
                elif event.key == pygame.K_LEFT and index > 0:
                    index -= 1
                    has_changed = True

                if has_changed:
                    features = x_test[index]
                    target_readable = y_test[index]

                    output = net.inference_feedforward(features)

                    print_output = [float(a[0]) for a in output]

                    success_rate = print_output[y_test[index]] * 100

                    draw.screen.fill(draw.WHITE)

                    draw.draw_input(x_test[index], draw.WIDTH//6, draw.HEIGHT//4, draw.WIDTH//2, is_RGB)

                    right_x = draw.WIDTH * 2 / 3 + 5
                    y_start = 40
                    line_spacing = 35

                    
                    
                    if dataset == "MNIST":
                        target_text = draw.font.render(f"Target: {target_readable}", True, (0, 0, 0))
                    else:
                        target_text = draw.font.render(f"Target: {target_readable} - {class_names[target_readable]}", True, (0, 0, 0))
                    draw.screen.blit(target_text, (draw.WIDTH//2-target_text.get_size()[0]//2, y_start))

                    if int(np.argmax(print_output)) == target_readable:
                        if dataset=="MNIST":
                            success_text = draw.small_font.render(f"Correctly predicted {target_readable} ({success_rate:.3f}%)", True, (0, 128, 0))
                        else:
                            success_text = draw.small_font.render(f"Correctly predicted {class_names[target_readable]} ({success_rate:.3f}%)", True, (0, 128, 0))
                    else:
                        if dataset=="MNIST":
                            success_text = draw.small_font.render(f"Wrongly predicted {np.argmax(print_output)} ({print_output[np.argmax(print_output)] * 100:.3f}%) instead of {target_readable} ({success_rate:.3f}%)", True, (128, 0, 0))
                        else:
                            success_text = draw.small_font.render(f"Wrongly predicted {class_names[np.argmax(print_output)]} ({print_output[np.argmax(print_output)] * 100:.3f}%) instead of {class_names[target_readable]} ({success_rate:.3f}%)", True, (128, 0, 0))
                    draw.screen.blit(success_text, (draw.WIDTH//2-success_text.get_size()[0]//2, y_start + line_spacing))

                    for i, prob in enumerate(print_output):
                        color = (0, 0, 0)
                        if i == int(np.argmax(print_output)):
                            color = (0, 0, 255)
                        if i == target_readable:
                            color = (200, 0, 0)

                        if dataset == "MNIST":
                            txt = draw.small_font.render(f"{i}: {prob:.3f}", True, color)
                        else:
                            txt = draw.small_font.render(f"{i}: {prob:.3f} - {class_names[i]}", True, color)
                        draw.screen.blit(txt, (right_x, y_start + 2*line_spacing + i * line_spacing))

                    pygame.display.update()
                    has_changed = False
                    
                    if find_incorrect:
                        if int(np.argmax(print_output)) == target_readable:
                            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT}))


    pygame.quit()



if __name__ == "__main__":

    import argparse

    from testing import import_params
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--params_folder", default="FINAL_PARAMS")
    parser.add_argument("--find_incorrect", default="0")
    parser.add_argument("--randomize_order", default="1")
    args = parser.parse_args()

    net = import_params.NN_from_params(args.dataset, args.params_folder)

    visual_test(net, args.dataset, args.find_incorrect=="1", args.randomize_order=="1")