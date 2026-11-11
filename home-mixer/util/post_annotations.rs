use std::collections::HashMap;
use std::time::Duration;
use thrift::protocol::{TInputProtocol, TOutputProtocol, TSerializable, TType};
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_strato::strato_thrift::{strato_decode, StratoResult};
use xai_vm_ranker_proto::experiment::{PostTopicEntity, PostTopics};

const FID_TOPICS: i16 = -9948;
const FID_ANNOTATIONS: i16 = 31524;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct PostAnnotations {
    pub topics: Vec<String>,
    pub entities: Vec<AnnotatedEntity>,
    pub normalized_tags: Vec<String>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct AnnotatedEntity {
    pub entity_id: i64,
    pub category_id: i64,
}

impl PostAnnotations {
    pub fn post_topics(&self) -> PostTopics {
        PostTopics {
            entities: self
                .entities
                .iter()
                .map(|e| PostTopicEntity {
                    entity_id: e.entity_id,
                    category_id: e.category_id,
                })
                .collect(),
            normalized_tags: self.normalized_tags.clone(),
        }
    }
}

pub async fn fetch_post_topics(
    strato_client: &(dyn StratoClient + Send + Sync),
    post_ids: &[u64],
    timeout: Duration,
) -> Option<HashMap<u64, PostTopics>> {
    if post_ids.is_empty() {
        return Some(HashMap::new());
    }
    let fetched = tokio::time::timeout(
        timeout,
        strato_client.batch_get_unified_post_annotations(post_ids),
    )
    .await
    .ok()?;
    Some(
        post_ids
            .iter()
            .zip(fetched)
            .filter_map(|(id, result)| {
                let bytes = result.ok()?;
                let annotations = decode_post_annotations(&bytes).ok()??;
                Some((*id, annotations.post_topics()))
            })
            .collect(),
    )
}

pub fn decode_post_annotations(bytes: &[u8]) -> Result<Option<PostAnnotations>, String> {
    match strato_decode::<UnifiedPostAnnotationsValue>(bytes) {
        Ok(StratoResult::Ok {
            value: Some(value), ..
        }) => Ok(Some(PostAnnotations {
            topics: value.topics,
            entities: value
                .annotations
                .entities
                .into_iter()
                .map(|e| AnnotatedEntity {
                    entity_id: e.qualified_id.entity_id,
                    category_id: e.category_id.entity_id,
                })
                .collect(),
            normalized_tags: value
                .annotations
                .tags
                .into_iter()
                .filter_map(|t| t.normalized_tag)
                .filter(|t| !t.is_empty())
                .collect(),
        })),
        Ok(StratoResult::Ok { value: None, .. }) | Ok(StratoResult::Err { .. }) => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

fn read_list<T>(
    i_prot: &mut dyn TInputProtocol,
    read: impl Fn(&mut dyn TInputProtocol) -> thrift::Result<T>,
) -> thrift::Result<Vec<T>> {
    let list = i_prot.read_list_begin()?;
    let mut out = Vec::with_capacity(list.size.max(0) as usize);
    for _ in 0..list.size {
        out.push(read(i_prot)?);
    }
    i_prot.read_list_end()?;
    Ok(out)
}

#[derive(Debug, Default, Clone)]
struct QualifiedId {
    entity_id: i64,
}

impl TSerializable for QualifiedId {
    fn read_from_in_protocol(i_prot: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        i_prot.read_struct_begin()?;
        let mut q = QualifiedId::default();
        loop {
            let field = i_prot.read_field_begin()?;
            if field.field_type == TType::Stop {
                break;
            }
            match field.id {
                Some(2) => q.entity_id = i_prot.read_i64()?,
                _ => i_prot.skip(field.field_type)?,
            }
            i_prot.read_field_end()?;
        }
        i_prot.read_struct_end()?;
        Ok(q)
    }

    fn write_to_out_protocol(&self, _o_prot: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        unimplemented!("decode-only")
    }
}

#[derive(Debug, Default, Clone)]
struct EntityWithMetadata {
    qualified_id: QualifiedId,
    category_id: QualifiedId,
}

impl TSerializable for EntityWithMetadata {
    fn read_from_in_protocol(i_prot: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        i_prot.read_struct_begin()?;
        let mut e = EntityWithMetadata::default();
        loop {
            let field = i_prot.read_field_begin()?;
            if field.field_type == TType::Stop {
                break;
            }
            match field.id {
                Some(1) => e.qualified_id = QualifiedId::read_from_in_protocol(i_prot)?,
                Some(3) => e.category_id = QualifiedId::read_from_in_protocol(i_prot)?,
                _ => i_prot.skip(field.field_type)?,
            }
            i_prot.read_field_end()?;
        }
        i_prot.read_struct_end()?;
        Ok(e)
    }

    fn write_to_out_protocol(&self, _o_prot: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        unimplemented!("decode-only")
    }
}

#[derive(Debug, Default, Clone)]
struct TagWithMetadata {
    normalized_tag: Option<String>,
}

impl TSerializable for TagWithMetadata {
    fn read_from_in_protocol(i_prot: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        i_prot.read_struct_begin()?;
        let mut t = TagWithMetadata::default();
        loop {
            let field = i_prot.read_field_begin()?;
            if field.field_type == TType::Stop {
                break;
            }
            match field.id {
                Some(3) if field.field_type == TType::String => {
                    t.normalized_tag = Some(i_prot.read_string()?)
                }
                _ => i_prot.skip(field.field_type)?,
            }
            i_prot.read_field_end()?;
        }
        i_prot.read_struct_end()?;
        Ok(t)
    }

    fn write_to_out_protocol(&self, _o_prot: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        unimplemented!("decode-only")
    }
}

#[derive(Debug, Default, Clone)]
struct UnifiedPostAnnotations {
    entities: Vec<EntityWithMetadata>,
    tags: Vec<TagWithMetadata>,
}

impl TSerializable for UnifiedPostAnnotations {
    fn read_from_in_protocol(i_prot: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        i_prot.read_struct_begin()?;
        let mut a = UnifiedPostAnnotations::default();
        loop {
            let field = i_prot.read_field_begin()?;
            if field.field_type == TType::Stop {
                break;
            }
            match field.id {
                Some(2) if field.field_type == TType::List => {
                    a.entities = read_list(i_prot, EntityWithMetadata::read_from_in_protocol)?;
                }
                Some(3) if field.field_type == TType::List => {
                    a.tags = read_list(i_prot, TagWithMetadata::read_from_in_protocol)?;
                }
                _ => i_prot.skip(field.field_type)?,
            }
            i_prot.read_field_end()?;
        }
        i_prot.read_struct_end()?;
        Ok(a)
    }

    fn write_to_out_protocol(&self, _o_prot: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        unimplemented!("decode-only")
    }
}

#[derive(Debug, Default, Clone)]
struct UnifiedPostAnnotationsValue {
    topics: Vec<String>,
    annotations: UnifiedPostAnnotations,
}

impl TSerializable for UnifiedPostAnnotationsValue {
    fn read_from_in_protocol(i_prot: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        i_prot.read_struct_begin()?;
        let mut tv = UnifiedPostAnnotationsValue::default();
        loop {
            let field = i_prot.read_field_begin()?;
            if field.field_type == TType::Stop {
                break;
            }
            match field.id {
                Some(FID_TOPICS) if field.field_type == TType::List => {
                    tv.topics = read_list(i_prot, |p| p.read_string())?;
                }
                Some(FID_ANNOTATIONS) if field.field_type == TType::Struct => {
                    tv.annotations = UnifiedPostAnnotations::read_from_in_protocol(i_prot)?;
                }
                _ => i_prot.skip(field.field_type)?,
            }
            i_prot.read_field_end()?;
        }
        i_prot.read_struct_end()?;
        Ok(tv)
    }

    fn write_to_out_protocol(&self, _o_prot: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        unimplemented!("decode-only")
    }
}
