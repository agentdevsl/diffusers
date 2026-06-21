<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DDIMVariantScheduler

[`DDIMVariantScheduler`] is a minimal, torch-only DDIM variant with a configurable `prediction_type`
(`"epsilon"`, `"sample"`, or `"v_prediction"`). It is behaviorally identical to [`DDIMScheduler`] and is
provided as a self-contained scheduler that follows the single-file scheduler convention.

## DDIMVariantScheduler
[[autodoc]] DDIMVariantScheduler

## DDIMVariantSchedulerOutput
[[autodoc]] schedulers.scheduling_ddim_variant.DDIMVariantSchedulerOutput
