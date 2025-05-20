<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

</head>
<body>
  <h1>Adversarial Example Generation and Model Fine-tuning with ResNet18</h1>

  <h2>Overview</h2>
  <p>This project explores the generation of <b>adversarial examples</b> and the fine-tuning of the <b>ResNet18 model</b> on the MNIST dataset. The main focus was on evaluating model performance under normal and adversarial conditions using techniques like <b>FGSM</b> (Fast Gradient Sign Method) and <b>Projected Gradient Descent (PGD)</b>.</p>

  <h2>Adversarial Example Techniques</h2>
  <p>Adversarial examples are inputs designed to deceive machine learning models by introducing small, imperceptible changes to the data. These changes cause models to misclassify the input, exposing weaknesses in model robustness.</p>
  <ul>
    <li><b>FGSM (Fast Gradient Sign Method)</b>: This method generates adversarial examples by perturbing the input data in the direction of the gradient of the loss function with respect to the input. The perturbation is scaled by a factor (epsilon), which controls the magnitude of the attack.</li>
    <li><b>Projected Gradient Descent (PGD)</b>: This is an iterative version of FGSM. It applies the gradient updates multiple times and projects the perturbations back into the allowed input space after each step, making it a more powerful adversarial attack.</li>
    <li><b>Other Techniques</b>: Other methods for generating adversarial examples include <b>Carlini-Wagner Attack</b> and <b>DeepFool</b>. These methods can be more sophisticated in terms of their ability to bypass defenses, but the focus in this project was on <b>FGSM</b> and <b>PGD</b>.</li>
  </ul>

  <h2>Model Fine-tuning with ResNet18 on MNIST</h2>
  <p>The <b>ResNet18 model</b> was fine-tuned on the <b>MNIST dataset</b> to evaluate performance on clean data and under adversarial conditions. Below are the results after fine-tuning for 1 epoch:</p>
  <ul>
    <li><b>Clean Accuracy</b>: 97.91%</li>
    <li><b>FGSM Accuracy</b>: 12.51%</li>
    <li><b>FGSM + Gaussian Accuracy</b>: 97.92%</li>
  </ul>
  <p>As seen in the results, <b>FGSM</b> significantly reduced the accuracy of the model, showing how vulnerable the ResNet18 model is to adversarial examples. However, when combined with Gaussian noise, the accuracy was restored, demonstrating the robustness of the model under certain defense mechanisms.</p>

  <h2>Projected Gradient Descent (PGD)</h2>
  <p><b>PGD</b> is an advanced adversarial attack method that refines adversarial examples iteratively. By applying small perturbations multiple times and ensuring that the perturbation remains within a feasible range, PGD generates more challenging adversarial examples compared to FGSM. This method is effective in evaluating model robustness and improving the adversarial training process.</p>

  <h2>Fine-tuning Results</h2>
  <p>The <b>ResNet18 model</b> was fine-tuned on the MNIST dataset and achieved the following performance metrics:</p>
  <pre>
    Fine-tuning pretrained ResNet on MNIST for 1 epoch...
    Epoch 1, Loss: 0.06693653437562896
    Fine-tuning complete.
    Model saved as 'finetuned_resnet18_mnist.pth'.

    Evaluation on 10000 MNIST samples (ResNet18):
    Clean Accuracy           : 9791/10000 = 97.91%
    FGSM Accuracy            : 1251/10000 = 12.51%
    FGSM + Gaussian Accuracy: 9792/10000 = 97.92%
  </pre>

  <h2>Model Evaluation</h2>
  <ul>
    <li><b>ResNet18 Performance</b>: The model achieved a clean accuracy of 97.91% on the MNIST test set, but its accuracy dropped significantly to 12.51% under <b>FGSM</b> adversarial attack. The model's performance improved to 97.92% when Gaussian noise was added to the adversarial examples.</li>
    <li><b>Adversarial Defense</b>: Incorporating noise or using advanced defense techniques like adversarial training can help improve the model's resilience to attacks like <b>FGSM</b> and <b>PGD</b>.</li>
  </ul>

  <h2>Technologies Used</h2>
  <ul>
    <li><b>ResNet18</b>: Pretrained model used for classification tasks on MNIST.</li>
    <li><b>FGSM and PGD</b>: Adversarial attack techniques for generating adversarial examples.</li>
    <li><b>Python</b>: Programming language used for model development and optimization.</li>
    <li><b>Pytorch</b>: Framework used for training the ResNet18 model and implementing adversarial attacks.</li>
    <li><b>TensorFlow</b>: For training and evaluating deep learning models.</li>
    <li><b>NumPy</b>: For numerical computations during the model training and evaluation process.</li>
  </ul>

  <h2>Future Improvements</h2>
  <ul>
    <li>Integrate more sophisticated adversarial defense mechanisms like <b>adversarial training</b> and <b>defensive distillation</b> to improve model robustness.</li>
    <li>Experiment with more complex models (e.g., <b>ResNet50</b>, <b>DenseNet</b>) to test their resistance to adversarial attacks.</li>
    <li>Explore other adversarial attack methods like <b>DeepFool</b> or <b>Carlini-Wagner</b> to better understand model vulnerabilities and enhance defenses.</li>
  </ul>

</body>
</html>
