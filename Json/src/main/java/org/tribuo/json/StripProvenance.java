/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tribuo.json;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.config.json.JsonProvenanceSerialization;
import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.primitives.HashProvenance;
import com.oracle.labs.mlrg.olcut.util.IOUtil;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.ensemble.EnsembleModel;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.EnsembleModelProvenance;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.EmptyDatasetProvenance;
import org.tribuo.provenance.impl.EmptyTrainerProvenance;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.tribuo.json.StripProvenance.ProvenanceTypes.ALL;
import static org.tribuo.json.StripProvenance.ProvenanceTypes.DATASET;
import static org.tribuo.json.StripProvenance.ProvenanceTypes.INSTANCE;
import static org.tribuo.json.StripProvenance.ProvenanceTypes.SYSTEM;
import static org.tribuo.json.StripProvenance.ProvenanceTypes.TRAINER;

/**
 * A main class for stripping out and storing provenance from a model.
 * <p>
 * Provenance stripping is useful for deploying models where others may
 * be able to inspect the model metadata and discover things about the model's
 * training procedure.
 */
public final class StripProvenance {
    private static final Logger logger = Logger.getLogger(StripProvenance.class.getName());

    private StripProvenance() {
    }

    /**
     * Types of provenance that can be removed.
     */
    public enum ProvenanceTypes {
        /**
         * Select the dataset provenance.
         */
        DATASET,
        /**
         * Select the trainer provenance.
         */
        TRAINER,
        /**
         * Select any instance provenance from the specific training run that created this model.
         */
        INSTANCE,
        /**
         * Selects any system information provenance.
         */
        SYSTEM,
        /**
         * Selects all provenance stored in the model.
         */
        ALL
    }

    /**
     * Creates a new model provenance with the requested provenances stripped out.
     *
     * @param old            The old model provenance.
     * @param provenanceHash The hash of the provenance (if requested it can be written into the new provenance for tracking).
     * @param opt            The program options.
     * @return A new model provenance.
     */
    private static ModelProvenance cleanProvenance(ModelProvenance old, String provenanceHash, StripProvenanceOptions opt) {
        // Dataset provenance
        DatasetProvenance datasetProvenance;
        if (opt.removeProvenances.contains(ALL) || opt.removeProvenances.contains(DATASET)) {
            datasetProvenance = new EmptyDatasetProvenance();
        } else {
            datasetProvenance = old.getDatasetProvenance();
        }
        // Trainer provenance
        TrainerProvenance trainerProvenance;
        if (opt.removeProvenances.contains(ALL) || opt.removeProvenances.contains(TRAINER)) {
            trainerProvenance = new EmptyTrainerProvenance();
        } else {
            trainerProvenance = old.getTrainerProvenance();
        }
        // Instance provenance
        OffsetDateTime time;
        Map<String, Provenance> instanceProvenance;
        if (opt.removeProvenances.contains(ALL) || opt.removeProvenances.contains(INSTANCE)) {
            instanceProvenance = new HashMap<>();
            time = OffsetDateTime.MIN;
        } else {
            instanceProvenance = new HashMap<>(old.getInstanceProvenance().getMap());
            time = old.getTrainingTime();
        }
        if (opt.storeHash) {
            logger.info("Writing provenance hash into instance map.");
            instanceProvenance.put("original-provenance-hash", new HashProvenance(opt.hashType, "original-provenance-hash", provenanceHash));
        }

        boolean stripSystem;
        if (opt.removeProvenances.contains(ALL) || opt.removeProvenances.contains(SYSTEM)) {
            stripSystem = true;
        } else {
            stripSystem = false;
        }

        return new ModelProvenance(old.getClassName(), time, datasetProvenance, trainerProvenance, instanceProvenance, !stripSystem);
    }

    /**
     * Creates a new ensemble provenance with the requested information removed.
     *
     * @param old              The old ensemble provenance.
     * @param memberProvenance The new member provenances.
     * @param provenanceHash   The old ensemble provenance hash.
     * @param opt              The program options.
     * @return The new ensemble provenance with the requested fields removed.
     */
    private static EnsembleModelProvenance cleanEnsembleProvenance(EnsembleModelProvenance old, ListProvenance<ModelProvenance> memberProvenance, String provenanceHash, StripProvenanceOptions opt) {
        // Dataset provenance
        DatasetProvenance datasetProvenance;
        if (opt.removeProvenances.contains(ALL) || opt.removeProvenances.contains(DATASET)) {
            datasetProvenance = new EmptyDatasetProvenance();
        } else {
            datasetProvenance = old.getDatasetProvenance();
        }
        // Trainer provenance
        TrainerProvenance trainerProvenance;
        if (opt.removeProvenances.contains(ALL) || opt.removeProvenances.contains(TRAINER)) {
            trainerProvenance = new EmptyTrainerProvenance();
        } else {
            trainerProvenance = old.getTrainerProvenance();
        }
        // Instance provenance
        OffsetDateTime time;
        Map<String, Provenance> instanceProvenance;
        if (opt.removeProvenances.contains(ALL) || opt.removeProvenances.contains(INSTANCE)) {
            instanceProvenance = new HashMap<>();
            time = OffsetDateTime.MIN;
        } else {
            instanceProvenance = new HashMap<>(old.getInstanceProvenance().getMap());
            time = old.getTrainingTime();
        }
        if (opt.storeHash) {
            logger.info("Writing provenance hash into instance map.");
            instanceProvenance.put("original-provenance-hash", new HashProvenance(opt.hashType, "original-provenance-hash", provenanceHash));
        }
        return new EnsembleModelProvenance(old.getClassName(), time, datasetProvenance, trainerProvenance, instanceProvenance, memberProvenance);
    }

    /**
     * Creates a copy of the old model with the requested provenance removed.
     *
     * @param oldModel       The model to remove provenance from.
     * @param provenanceHash A hash of the old provenance.
     * @param opt            The program options.
     * @param <T>            The output type.
     * @return A copy of the model with redacted provenance.
     * @throws InvocationTargetException If the model doesn't expose a copy method (all models should do).
     * @throws IllegalAccessException    If the model's copy method is not accessible.
     * @throws NoSuchMethodException     If the model's copy method isn't present.
     */
    @SuppressWarnings("unchecked") // cast of model after call to copy which returns model.
    private static <T extends Output<T>> ModelTuple<T> convertModel(Model<T> oldModel, String provenanceHash, StripProvenanceOptions opt) throws InvocationTargetException, IllegalAccessException, NoSuchMethodException {
        if (oldModel instanceof EnsembleModel) {
            EnsembleModelProvenance oldProvenance = ((EnsembleModel<T>) oldModel).getProvenance();
            List<ModelProvenance> newProvenances = new ArrayList<>();
            List<Model<T>> newModels = new ArrayList<>();
            for (Model<T> e : ((EnsembleModel<T>) oldModel).getModels()) {
                ModelTuple<T> tuple = convertModel(e, provenanceHash, opt);
                newProvenances.add(tuple.provenance);
                newModels.add(tuple.model);
            }
            ListProvenance<ModelProvenance> listProv = new ListProvenance<>(newProvenances);
            EnsembleModelProvenance cleanedProvenance = cleanEnsembleProvenance(oldProvenance, listProv, provenanceHash, opt);
            Class<? extends Model> clazz = oldModel.getClass();
            Method copyMethod = clazz.getDeclaredMethod("copy", String.class, ModelProvenance.class, List.class);
            boolean accessible = copyMethod.isAccessible();
            copyMethod.setAccessible(true);
            String newName = oldModel.getName().isEmpty() ? "deprovenanced" : oldModel.getName() + "-deprovenanced";
            EnsembleModel<T> output = (EnsembleModel<T>) copyMethod.invoke(oldModel, newName, cleanedProvenance, newModels);
            copyMethod.setAccessible(accessible);
            return new ModelTuple<>(output, cleanedProvenance);
        } else {
            ModelProvenance oldProvenance = oldModel.getProvenance();
            ModelProvenance cleanedProvenance = cleanProvenance(oldProvenance, provenanceHash, opt);
            Class<? extends Model> clazz = oldModel.getClass();
            Method copyMethod = clazz.getDeclaredMethod("copy", String.class, ModelProvenance.class);
            boolean accessible = copyMethod.isAccessible();
            copyMethod.setAccessible(true);
            String newName = oldModel.getName().isEmpty() ? "deprovenanced" : oldModel.getName() + "-deprovenanced";
            Model<T> output = (Model<T>) copyMethod.invoke(oldModel, newName, cleanedProvenance);
            copyMethod.setAccessible(accessible);
            return new ModelTuple<>(output, cleanedProvenance);
        }
    }

    /**
     * Command line options.
     */
    public static class StripProvenanceOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "A program for removing Provenance information from a Tribuo Model or SequenceModel.";
        }

        /**
         * Stores a hash of the model provenance in the stripped model.
         */
        @Option(charName = 'h', longName = "store-provenance-hash", usage = "Stores a hash of the model provenance in the stripped model.")
        public boolean storeHash;
        /**
         * The model to load.
         */
        @Option(charName = 'i', longName = "input-model-path", usage = "The model to load.")
        public File inputModel;
        /**
         * The location to write out the stripped model.
         */
        @Option(charName = 'o', longName = "output-model-path", usage = "The location to write out the stripped model.")
        public File outputModel;
        /**
         * Write out the stripped provenance as json.
         */
        @Option(charName = 'p', longName = "provenance-path", usage = "Write out the stripped provenance as json.")
        public File provenanceFile;
        /**
         * The provenances to remove
         */
        @Option(charName = 'r', longName = "remove-provenances", usage = "The provenances to remove")
        public EnumSet<ProvenanceTypes> removeProvenances = EnumSet.noneOf(ProvenanceTypes.class);
        /**
         * The hash type to use.
         */
        @Option(charName = 't', longName = "hash-type", usage = "The hash type to use.")
        public ProvenanceUtil.HashType hashType = ObjectProvenance.DEFAULT_HASH_TYPE;
        /**
         * Read and write protobuf formatted models.
         */
        @Option(longName = "model-protobuf", usage = "Read and write protobuf formatted models.")
        public boolean protobuf;
    }

    /**
     * Runs StripProvenance.
     *
     * @param args the command line arguments
     * @param <T>  The {@link Output} subclass.
     */
    public static <T extends Output<T>> void main(String[] args) {

        //
        // Use the labs format logging.
        LabsLogFormatter.setAllLogFormatters();

        StripProvenanceOptions o = new StripProvenanceOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args, o);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }

        if (o.inputModel == null || o.outputModel == null) {
            logger.info(cm.usage());
            System.exit(1);
        }

        try {
            logger.info("Loading model from " + o.inputModel);
            Model<?> model;
            if (o.protobuf) {
                model = Model.deserializeFromFile(o.inputModel.toPath());
            } else {
                try (ObjectInputStream ois = IOUtil.getObjectInputStream(o.inputModel)) {
                    model = (Model<?>) ois.readObject();
                }
            }

            ModelProvenance oldProvenance = model.getProvenance();

            logger.info("Marshalling provenance and creating JSON.");
            JsonProvenanceSerialization jsonProvenanceSerialization = new JsonProvenanceSerialization(true);
            String jsonResult = jsonProvenanceSerialization.marshalAndSerialize(oldProvenance);

            logger.info("Hashing JSON file");
            MessageDigest digest = o.hashType.getDigest();
            byte[] digestBytes = digest.digest(jsonResult.getBytes(StandardCharsets.UTF_8));
            String provenanceHash = ProvenanceUtil.bytesToHexString(digestBytes);
            logger.info("Provenance hash = " + provenanceHash);

            if (o.provenanceFile != null) {
                logger.info("Writing JSON provenance to " + o.provenanceFile.toString());
                try (PrintWriter writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(o.provenanceFile), StandardCharsets.UTF_8))) {
                    writer.println(jsonResult);
                }
            }

            ModelTuple<?> tuple = convertModel(model, provenanceHash, o);
            logger.info("Writing model to " + o.outputModel);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(o.outputModel))) {
                oos.writeObject(tuple.model);
            }

            ModelProvenance newProvenance = tuple.provenance;
            logger.info("Marshalling provenance and creating JSON.");
            String newJsonResult = jsonProvenanceSerialization.marshalAndSerialize(newProvenance);

            logger.info("Old provenance = \n" + jsonResult);
            logger.info("New provenance = \n" + newJsonResult);
        } catch (NoSuchMethodException e) {
            logger.log(Level.SEVERE, "Model.copy method missing on a class which extends Model.", e);
        } catch (IllegalAccessException e) {
            logger.log(Level.SEVERE, "Failed to modify protection on inner copy method on Model.", e);
        } catch (InvocationTargetException e) {
            logger.log(Level.SEVERE, "Failed to invoke inner copy method on Model.", e);
        } catch (UnsupportedEncodingException e) {
            logger.log(Level.SEVERE, "Unsupported encoding exception.", e);
        } catch (FileNotFoundException e) {
            logger.log(Level.SEVERE, "Failed to find the input file.", e);
        } catch (IOException e) {
            logger.log(Level.SEVERE, "IO error when reading or writing a file.", e);
        } catch (ClassNotFoundException e) {
            logger.log(Level.SEVERE, "The model and/or provenance classes are not on the classpath.", e);
        }

    }

    /**
     * It's a record. Or at least it will be.
     *
     * @param <T> The output type.
     */
    private static class ModelTuple<T extends Output<T>> {
        public final Model<T> model;
        public final ModelProvenance provenance;

        /**
         * Constructs a model tuple.
         *
         * @param model      The model.
         * @param provenance The provenance.
         */
        public ModelTuple(Model<T> model, ModelProvenance provenance) {
            this.model = model;
            this.provenance = provenance;
        }
    }
}
