import matplotlib.pyplot as plt
from utils.configs import configs

with open(configs.training_log_filename,  "r") as file:
    log_list = eval(file.readline())
figure, axis = plt.subplots(3)
figure.suptitle('Scenario: {}'.format(configs.scenario))
axis[0].plot(log_list[0])
axis[0].set_title("Training Loss")
axis[0].set_ylabel('Loss')
axis[0].grid()
axis[1].plot(log_list[1])
axis[1].set_title("Validation Loss")
axis[1].set_ylabel('Loss')
axis[1].grid()
axis[2].plot(log_list[2], label='Training')
axis[2].plot(log_list[3], label='Validation')
axis[2].set_title("Training and Validation EER (%)")
axis[2].set_xlabel('Epochs')
axis[2].set_ylabel('EER (%)')
axis[2].legend()
axis[2].grid()
# plt.show()
plt.savefig(configs.training_log_plot_filename)