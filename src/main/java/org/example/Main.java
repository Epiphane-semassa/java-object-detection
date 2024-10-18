package org.example;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;


public class Main {
    public static void main(String[] args) throws IOException, MalformedModelException, ModelNotFoundException {
        System.out.println("Hello world Java object detection demo!".toUpperCase());

        Path imagePath = Paths.get("C:\\Users\\Lenovo\\Downloads\\81b855ddedc6ca9e1974405ee4f1c8a9.jpg");
        Image img = ImageFactory.getInstance().fromFile(imagePath);

        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .optApplication(Application.CV.OBJECT_DETECTION)
                .setTypes(Image.class, DetectedObjects.class)
                .optFilter("backbone", "resnet50")
                .optProgress(new ProgressBar())
                .build();

        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detection = predictor.predict(img);
                System.out.println("objects data : " + detection);
                System.out.println("Objects detected : " + detection.getNumberOfObjects());
                System.out.println(detection.toJson());
            } catch (TranslateException e) {
                throw new RuntimeException(e);
            }
        }
    }
}